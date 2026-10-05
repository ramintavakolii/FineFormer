"""
Unified entry point for Region / Time / Hybrid training.

Preferred CLI:
  python main.py --model region --task HS --cycle 1
  python main.py --model time --task HB --cycle 1 --lr 1e-4 --dropout 0.14
  python main.py --model hybrid --task BS --cycle 2 --lr 2e-4 --dropout 0.10

Legacy Hydra-style overrides are still accepted when argv does not use --flags:
  python main.py model=region task=hs cycle=cycle_1
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datamodule import DataModule
from src.models.factory import build_model
from src.training.cross_validator import CrossValidator
from src.training.cyclic_manager import prepare_fold_transfer, run_save_path
from src.utils.seeding import seed_everything


def _drop_nulls(node):
    """Return a plain dict with null/None entries removed (recursive for dicts)."""
    if isinstance(node, DictConfig):
        node = OmegaConf.to_container(node, resolve=True)
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if v is None:
            continue
        if isinstance(v, dict):
            cleaned = _drop_nulls(v)
            if cleaned:
                out[k] = cleaned
        else:
            out[k] = v
    return out


def resolve_effective_run(cfg: DictConfig) -> DictConfig:
    """Select the single task×cycle training block (avoids duplicate keys)."""
    task_name = cfg.task.name
    block = cfg.cycle.tasks[task_name]
    effective = OmegaConf.create(OmegaConf.to_container(block, resolve=True))
    for key in ("training", "search", "augmentation"):
        if key in cfg and cfg[key] is not None:
            incoming = _drop_nulls(cfg[key])
            if incoming:
                effective[key] = OmegaConf.merge(effective[key], OmegaConf.create(incoming))
    return effective


def print_effective_config(cfg: DictConfig, effective: DictConfig, model_params: dict):
    print("\n" + "=" * 70)
    print("EFFECTIVE CONFIGURATION")
    print("=" * 70)
    print(f"model: {cfg.model.name} (notebook type={cfg.model.type})")
    print(f"task:  {cfg.task.name} ({cfg.task.task_type})")
    print(f"cycle: {cfg.cycle.cycle}")
    print(f"device: {cfg.device}")
    print(f"seed: {cfg.seed} (null = historical-like)")
    print(f"results_root: {cfg.output.results_root}")
    print(f"dataset_path: {cfg.data.dataset_path}")
    print(f"fold_indices_path: {cfg.data.fold_indices_path}")
    print(f"use_synthetic: {cfg.data.use_synthetic}")
    print("\n-- training --")
    print(OmegaConf.to_yaml(effective.training))
    print("-- search --")
    print(OmegaConf.to_yaml(effective.search))
    print("-- augmentation --")
    print(OmegaConf.to_yaml(effective.augmentation))
    print("-- model_params --")
    for k, v in model_params.items():
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")


def iter_hyperparams(effective: DictConfig, hyperparam_search: bool):
    train = effective.training
    if not hyperparam_search:
        yield (
            float(train.learning_rate),
            float(train.dropout),
            float(train.weight_decay),
        )
        return

    search = effective.search
    lrs = list(search.learning_rate)
    drops = list(search.dropout)
    wds = list(search.weight_decay)
    for lr, dropout, wd in itertools.product(lrs, drops, wds):
        yield float(lr), float(dropout), float(wd)


def run_one(cfg: DictConfig, effective: DictConfig, dm: DataModule,
            lr: float, dropout: float, weight_decay: float):
    train = effective.training
    aug = effective.augmentation
    task_name = cfg.task.name
    cycle = int(cfg.cycle.cycle)
    task_type = cfg.task.task_type
    model_name = cfg.model.name

    model_params = {
        "n_rois": int(dm.X_healthy.shape[2]),
        "n_timesteps": int(dm.X_healthy.shape[1]),
        "d_model": int(cfg.model.d_model),
        "n_heads": int(cfg.model.n_heads),
        "n_layers": int(cfg.model.n_layers),
        "n_classes": int(cfg.model.n_classes),
        "dropout": float(dropout),
        "type": str(cfg.model.type),
    }

    print_effective_config(cfg, effective, model_params)

    probe = build_model(model_name, **model_params)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"model parameter count: {n_params:,}")
    print(f"input shape: (B, {model_params['n_timesteps']}, {model_params['n_rois']})")
    print(f"optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")
    print(f"dropout: {dropout}")
    print(f"warmup_epochs: {train.warmup_epochs}")
    print(f"scheduler: ReduceLROnPlateau(mode=max, factor={train.scheduler_factor}, patience=10)")
    del probe

    save_path = run_save_path(cfg.output.results_root, model_name, task_name, cycle)
    os.makedirs(save_path, exist_ok=True)

    fold_id_base = int(train.fold_id_base)
    if cfg.run.folds is None:
        fold_indices = list(range(cfg.data.n_splits))
    else:
        fold_indices = list(cfg.run.folds)

    if bool(train.fine_tune):
        for fold in fold_indices:
            fold_num = fold + fold_id_base

            prepare_fold_transfer(
                cfg.output.results_root, model_name, task_name, cycle, fold_num
            )

    cv = CrossValidator(
        X_healthy=dm.X_healthy,
        X_schizo=dm.X_schizo,
        X_bipolar=dm.X_bipolar,
        fold_indices_path=cfg.data.fold_indices_path,
        device=torch.device(cfg.device),
        model_class=lambda **kw: build_model(model_name, **kw),
        scheduler_factor=float(train.scheduler_factor),
        model_params=model_params,
        n_splits=int(cfg.data.n_splits),
        batch_size=int(train.batch_size),
        num_epochs=int(train.num_epochs),
        patience=int(train.patience),
        random_state=int(cfg.data.fold_random_state),
        initial_lr=lr,
        fine_tune=bool(train.fine_tune),
        reinit_classifier=bool(train.reinit_classifier),
        freeze_encoder=bool(train.freeze_encoder),
        weight_decay=weight_decay,
        class_names=list(cfg.task.class_names),
        save_path=save_path
    )

    metrics, summary = cv.run_task(task_type=task_type, fold_indices=fold_indices)
    metrics, summary = cv.run_task(task_type=task_type, fold_indices=fold_indices)
    return metrics, summary


def run(cfg: DictConfig):
    seed_everything(cfg.seed, deterministic=bool(cfg.deterministic))

    if str(cfg.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda was requested but CUDA is not available. "
            "Use --device cpu or install a CUDA-capable PyTorch build."
        )

    effective = resolve_effective_run(cfg)

    dm = DataModule(
        dataset_path=cfg.data.dataset_path,
        fold_indices_path=cfg.data.fold_indices_path,
        fold_random_state=int(cfg.data.fold_random_state),
        n_splits=int(cfg.data.n_splits),
        use_synthetic=bool(cfg.data.use_synthetic),
    ).setup()

    print(f"dataset sizes: H={len(dm.X_healthy)} S={len(dm.X_schizo)} B={len(dm.X_bipolar)}")

    all_summaries = []

    for lr, dropout, wd in iter_hyperparams(effective, bool(cfg.run.hyperparam_search)):
        print(f"\n===== Training with LR={lr}, Dropout={dropout}, Weight Decay={wd} =====")
        OmegaConf.set_struct(cfg, False)
        cfg.model.dropout = dropout
        OmegaConf.set_struct(cfg, True)

        effective.training.learning_rate = lr
        effective.training.dropout = dropout
        effective.training.weight_decay = wd

        _, summary = run_one(cfg, effective, dm, lr, dropout, wd)
        all_summaries.append({"lr": lr, "dropout": dropout, "weight_decay": wd, "summary": summary})

    if cfg.run.hyperparam_search and all_summaries:
        best = max(
            all_summaries,
            key=lambda x: x["summary"].get("balanced_accuracy", {}).get("mean", 0.0),
        )
        print("\n" + "=" * 60)
        print("BEST HYPERPARAMETERS BASED ON BALANCED ACCURACY:")
        print(f"LR: {best['lr']}, Dropout: {best['dropout']}, Weight Decay: {best['weight_decay']}")
        print("=" * 60)

    return all_summaries


def _normalize_cycle_override(overrides: list[str]) -> list[str]:
    """Map cycle=cycle_N → cycle={model}/cycle_N when the model is known."""
    model = None
    cycle = None
    for item in overrides:
        if item.startswith("model="):
            model = item.split("=", 1)[1]
        elif item.startswith("cycle="):
            cycle = item.split("=", 1)[1]
    if model in {"region", "time", "hybrid"} and cycle in {"cycle_1", "cycle_2"}:
        return [
            f"cycle={model}/{cycle}" if item.startswith("cycle=") else item
            for item in overrides
        ]
    return overrides


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose Hydra config from YAML + overrides (no hydra.main; Py3.14-safe)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    overrides = _normalize_cycle_override(list(overrides or []))
    GlobalHydra.instance().clear()
    config_dir = str(ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="fMRI Transformer training (Region / Time / Hybrid)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=["region", "time", "hybrid"],
                   help="Architecture")
    p.add_argument("--task", required=True, choices=["HS", "HB", "BS", "hs", "hb", "bs"],
                   help="Classification task")
    p.add_argument("--cycle", required=True, type=int, choices=[1, 2],
                   help="Transfer cycle")
    p.add_argument("--lr", type=float, default=None,
                   help="Override training.learning_rate")
    p.add_argument("--dropout", type=float, default=None,
                   help="Override model/training dropout")
    # Useful high-level knobs (optional)
    p.add_argument("--seed", type=int, default=None,
                   help="Global seed (omit for historical-like unseeded mode)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"],
                   help="Override device")
    p.add_argument("--folds", default=None,
                   help="Comma-separated 0-based fold indices, e.g. 0 or 0,1")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Explicit synthetic smoke-test data only. "
            "Never used automatically; missing real data raises an error."
        ),
    )
    p.add_argument("--hyperparam-search", action="store_true",
                   help="Sweep search grids from cycle×task YAML")
    p.add_argument("--num-epochs", "--epochs", type=int, default=None, dest="num_epochs",
                   help="Override training.num_epochs (debugging)")
    return p


def cli_to_overrides(args: argparse.Namespace) -> list[str]:
    task = args.task.lower()
    overrides = [
        f"model={args.model}",
        f"task={task}",
        f"cycle={args.model}/cycle_{args.cycle}",
    ]
    if args.lr is not None:
        overrides.append(f"training.learning_rate={args.lr}")
    if args.dropout is not None:
        overrides.append(f"training.dropout={args.dropout}")
        overrides.append(f"model.dropout={args.dropout}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.device is not None:
        overrides.append(f"device={args.device}")
    if args.synthetic:
        overrides.append("data.use_synthetic=true")
    if args.hyperparam_search:
        overrides.append("run.hyperparam_search=true")
    if args.num_epochs is not None:
        overrides.append(f"training.num_epochs={args.num_epochs}")
    if args.folds is not None:
        folds = [int(x.strip()) for x in str(args.folds).split(",") if x.strip() != ""]
        overrides.append(f"run.folds=[{','.join(str(f) for f in folds)}]")
    return overrides


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Preferred: argparse flags. Fallback: raw Hydra overrides.
    if not argv or argv[0].startswith("-"):
        parser = build_arg_parser()
        args = parser.parse_args(argv)
        overrides = cli_to_overrides(args)
    else:
        overrides = argv

    cfg = load_config(overrides)

    return run(cfg)


if __name__ == "__main__":
    main()
