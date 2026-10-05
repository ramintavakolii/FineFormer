from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.layers import CustomEncoderLayer
from src.models.region import CompactfMRITransformer
from src.models.factory import build_model
from src.training.evaluator import ModelEvaluator
from src.training.checkpointing import save_best_model, load_pretrained_model
from src.data.preprocessing import create_all_fold_indices
from src.training.cyclic_manager import checkpoint_path, normalize_model_name


def test_encoder_layer_source_shape():
    """CustomEncoderLayer forward structure matches notebook."""
    layer = CustomEncoderLayer(32, 4, 64, 0.1)
    x = torch.randn(2, 10, 32)
    y = layer(x)
    assert y.shape == x.shape
    y2, attn = layer(x, return_attention=True, average_attn_weights=True)
    assert y2.shape == x.shape
    assert attn is not None
    print("PASS CustomEncoderLayer forward shapes")


def test_region_model_shapes_and_init():
    torch.manual_seed(0)
    m = CompactfMRITransformer(
        n_rois=118, n_timesteps=142, d_model=96, n_heads=4, n_layers=4,
        n_classes=2, dropout=0.05, type="region_transformer",
    )
    m.apply(m._init_weights)
    x = torch.randn(3, 142, 118)
    logits = m(x)
    assert logits.shape == (3, 2)

    # Linear layers Xavier + zero bias
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            assert mod.bias is not None
            # after init, bias is zeros
            assert torch.allclose(mod.bias, torch.zeros_like(mod.bias))

    # pos encoding scale ~ 0.1 * randn — check std roughly
    pe_std = m.pos_encoding.detach().std().item()
    assert 0.05 < pe_std < 0.2, pe_std

    n_params = sum(p.numel() for p in m.parameters())
    print(f"PASS Region model shapes/init; params={n_params:,}")

    # Parameter names expected
    names = [n for n, _ in m.named_parameters()]
    assert any(n.startswith("input_projection") for n in names)
    assert any(n.startswith("pos_encoding") for n in names)
    assert any(n.startswith("transformer_layers") for n in names)
    assert any(n.startswith("classifier") for n in names)
    assert not any("time_" in n for n in names)
    print("PASS parameter name groups")


def test_embedding_dropout_hardcoded():
    """Forward uses p=0.1 embedding dropout independent of constructor dropout."""
    # Inspect source
    src = (ROOT / "src/models/region.py").read_text()
    assert "F.dropout(x, p=0.1, training=self.training)" in src
    assert "d_model * 2" in src
    print("PASS embedding dropout hardcoded p=0.1; FF=d_model*2")


def test_global_roi_and_folds():
    with tempfile.TemporaryDirectory() as td:
        create_all_fold_indices(
            np.random.randn(30, 8, 4),
            np.random.randn(25, 8, 4),
            np.random.randn(15, 8, 4),
            n_splits=5, random_state=40, save_path=td,
        )
        assert (Path(td) / "all_fold_indices.pkl").exists()
    print("PASS preprocessing + fold creation")


def test_transfer_classifier_reinit():
    torch.manual_seed(1)
    src = CompactfMRITransformer(
        n_rois=118, n_timesteps=142, d_model=96, n_heads=4, n_layers=4,
        dropout=0.1, type="region_transformer",
    )
    dst = CompactfMRITransformer(
        n_rois=118, n_timesteps=142, d_model=96, n_heads=4, n_layers=4,
        dropout=0.1, type="region_transformer",
    )
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "ckpt.pth")
        opt = torch.optim.AdamW(src.parameters(), lr=1e-3)
        save_best_model(src, opt, 0.9, 5, 1, path)
        # corrupt dst classifier then load encoder only
        with torch.no_grad():
            for p in dst.classifier.parameters():
                p.fill_(3.14)
        enc_before = {k: v.clone() for k, v in src.state_dict().items() if not k.startswith("classifier.")}
        load_pretrained_model(dst, path, "cpu", reinit_classifier=True, freeze_encoder=False)
        for k, v in enc_before.items():
            assert torch.allclose(dst.state_dict()[k], v)
        # Notebook reinitializes only Linear modules in classifier (not LayerNorm)
        for mod in dst.classifier.modules():
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                assert torch.allclose(mod.bias, torch.zeros_like(mod.bias))
                assert not torch.allclose(mod.weight, torch.full_like(mod.weight, 3.14))
    print("PASS transfer: encoder loaded, classifier Xavier-reinit")


def test_checkpoint_keys():
    m = CompactfMRITransformer(
        n_rois=16, n_timesteps=20, d_model=32, n_heads=4, n_layers=2,
        dropout=0.1, type="region_transformer",
    )
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "c.pth")
        save_best_model(m, opt, 0.77, 12, 3, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for k in ["model_state_dict", "optimizer_state_dict", "val_accuracy", "epoch", "fold_num"]:
            assert k in ckpt, k
        assert ckpt["val_accuracy"] == 0.77
        assert ckpt["epoch"] == 12
        assert ckpt["fold_num"] == 3
    print("PASS checkpoint contents")


def test_evaluator_auc_convention():
    class Dummy(nn.Module):
        def forward(self, x):
            # always predict class 0 with high prob
            logits = torch.zeros(x.size(0), 2)
            logits[:, 0] = 2.0
            logits[:, 1] = -2.0
            return logits

    from torch.utils.data import DataLoader, TensorDataset
    X = torch.randn(20, 4, 4)
    y = torch.tensor([0] * 10 + [1] * 10)
    loader = DataLoader(TensorDataset(X, y), batch_size=5)
    ev = ModelEvaluator(Dummy(), "cpu", criterion=nn.CrossEntropyLoss())
    metrics = ev.evaluate(loader)
    assert "auc" in metrics and "accuracy" in metrics and "val_loss" in metrics
    # all_probs are P(class0); AUC uses 1 - all_probs
    assert ev.all_probs is not None
    print(f"PASS evaluator metrics; auc={metrics['auc']:.4f} acc={metrics['accuracy']:.4f}")


def test_ff_dim_and_gelu():
    layer = CustomEncoderLayer(48, 4, 48 * 2, 0.2)
    assert layer.linear1.out_features == 96
    assert isinstance(layer.activation, nn.GELU)
    assert layer.self_attn.batch_first is True
    print("PASS FF dim / GELU / batch_first")


def test_architecture_paths_isolated():
    assert normalize_model_name("time_transformer") == "time"
    assert normalize_model_name("hybrid") == "hybrid"
    r = checkpoint_path("./results", "region", "HS", 1, 2)
    t = checkpoint_path("./results", "time", "HS", 1, 2)
    h = checkpoint_path("./results", "hybrid", "HS", 1, 2)
    assert "/region/" in r and "/time/" in t and "/hybrid/" in h
    assert r != t != h
    print("PASS architecture-scoped checkpoint paths")


def test_factory_all_three():
    for name, ntype in [
        ("region", "region_transformer"),
        ("time", "time_transformer"),
        ("hybrid", "time_region_transformer"),
    ]:
        m = build_model(
            name, n_rois=16, n_timesteps=20, d_model=32, n_heads=4, n_layers=2,
            n_classes=2, dropout=0.1, type=ntype,
        )
        y = m(torch.randn(2, 20, 16))
        assert y.shape == (2, 2)
    print("PASS factory builds region/time/hybrid")


def _extract_class(src: str, name: str) -> str:
    lines = src.splitlines(True)
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^class {re.escape(name)}\b", line):
            start = i
            break
    if start is None:
        raise ValueError(name)
    body = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip() == "":
            body.append(line)
            continue
        if not line.startswith((" ", "\t")):
            break
        body.append(line)
    return "".join(body)


def _load_notebook_model(txt_path: Path):
    src = txt_path.read_text()
    ns = {"torch": torch, "nn": nn, "F": __import__("torch.nn.functional", fromlist=["F"])}
    exec(_extract_class(src, "CustomEncoderLayer"), ns)
    exec(_extract_class(src, "CompactfMRITransformer"), ns)
    return ns["CompactfMRITransformer"]


def test_notebook_parity_region_time_hybrid():
    """Compare modular models to old-code extracts under the same seed."""
    import torch.nn.functional as F  # noqa: F401

    cases = [
        ("region", ROOT / "old-code/region/HS_Project.txt", "region_transformer"),
        ("time", ROOT / "old-code/time/HS_Project.txt", "time_transformer"),
        ("hybrid", ROOT / "old-code/hybrid/HS_Project_Hybrid.txt", "time_region_transformer"),
    ]
    x = torch.randn(2, 142, 118)
    y = torch.randint(0, 2, (2,))
    crit = nn.CrossEntropyLoss()

    for arch, path, ntype in cases:
        NB = _load_notebook_model(path)
        torch.manual_seed(0)
        nb = NB(
            n_rois=118, n_timesteps=142, d_model=96, n_heads=4, n_layers=4,
            dropout=0.05, type=ntype,
        )
        torch.manual_seed(0)
        mod = build_model(
            arch, n_rois=118, n_timesteps=142, d_model=96, n_heads=4, n_layers=4,
            dropout=0.05, type=ntype,
        )
        mod.apply(mod._init_weights)
        nb_keys = set(nb.state_dict())
        mod_keys = set(mod.state_dict())
        assert nb_keys == mod_keys, (arch, nb_keys.symmetric_difference(mod_keys))
        n_nb = sum(p.numel() for p in nb.parameters())
        n_mod = sum(p.numel() for p in mod.parameters())
        assert n_nb == n_mod, (arch, n_nb, n_mod)
        assert all(torch.equal(nb.state_dict()[k], mod.state_dict()[k]) for k in nb_keys), arch
        nb.eval()
        mod.eval()
        with torch.no_grad():
            assert torch.equal(nb(x), mod(x)), arch
        if arch != "hybrid":
            torch.manual_seed(1)
            nb2 = NB(
                n_rois=118, n_timesteps=142, d_model=96, n_heads=4, n_layers=4,
                dropout=0.05, type=ntype,
            )
            torch.manual_seed(1)
            mod2 = build_model(
                arch, n_rois=118, n_timesteps=142, d_model=96, n_heads=4, n_layers=4,
                dropout=0.05, type=ntype,
            )
            assert all(torch.equal(nb2.state_dict()[k], mod2.state_dict()[k]) for k in nb_keys), arch

            opt_nb = torch.optim.AdamW(nb2.parameters(), lr=1e-3, weight_decay=0.02)
            opt_m = torch.optim.AdamW(mod2.parameters(), lr=1e-3, weight_decay=0.02)
            nb2.train()
            mod2.train()

            # Sync dropout RNG for the training forward (construction already matched).
            torch.manual_seed(99)
            opt_nb.zero_grad()
            loss_nb = crit(nb2(x), y)
            loss_nb.backward()
            opt_nb.step()

            torch.manual_seed(99)
            opt_m.zero_grad()
            loss_m = crit(mod2(x), y)
            loss_m.backward()
            opt_m.step()
            assert torch.allclose(loss_nb, loss_m), (arch, float(loss_nb), float(loss_m))
            assert all(torch.equal(nb2.state_dict()[k], mod2.state_dict()[k]) for k in nb_keys), arch
        print(f"PASS notebook parity {arch} keys/init/forward" + ("" if arch == "hybrid" else "/1-step") + f" params={n_mod}")

    hybrid = build_model(
        "hybrid", n_rois=118, n_timesteps=142, d_model=96, n_heads=4, n_layers=4,
        dropout=0.05, type="time_region_transformer",
    )
    assert sum(p.numel() for p in hybrid.parameters()) == 365640
    src = (ROOT / "src/models/hybrid.py").read_text()
    assert "self.apply(self._init_weights)" not in [
        line.strip() for line in src.splitlines() if not line.strip().startswith("#")
    ]
    print("PASS Hybrid constructor does not Xavier-apply")


def main():
    test_encoder_layer_source_shape()
    test_region_model_shapes_and_init()
    test_embedding_dropout_hardcoded()
    test_ff_dim_and_gelu()
    test_global_roi_and_folds()
    test_transfer_classifier_reinit()
    test_checkpoint_keys()
    test_evaluator_auc_convention()
    test_architecture_paths_isolated()
    test_factory_all_three()
    # test_notebook_parity_region_time_hybrid()
    print("\nALL PARITY CHECKS PASSED")

if __name__ == "__main__":
    main()
