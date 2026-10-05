import os
import shutil
from typing import Optional, Tuple

TRANSFER_GRAPH = {
    ("HB", 1): ("HS", 1),
    ("BS", 1): ("HB", 1),
    ("HS", 2): ("BS", 1),
    ("HB", 2): ("HS", 2),
    ("BS", 2): ("HB", 2),
}

TASK_TYPE = {
    "HS": "healthy_vs_schizo",
    "HB": "healthy_vs_bipolar",
    "BS": "bipolar_vs_schizo",
}


def normalize_model_name(model: str) -> str:
    name = str(model).lower().replace("-", "_")
    aliases = {
        "region_transformer": "region",
        "time_transformer": "time",
        "time_region_transformer": "hybrid",
        "hybrid_transformer": "hybrid",
        "time_region": "hybrid",
    }
    return aliases.get(name, name)


def resolve_source(task: str, cycle: int) -> Optional[Tuple[str, int]]:
    return TRANSFER_GRAPH.get((task.upper(), int(cycle)))


def _model_root(results_root: str, model: str, task: str, cycle: int) -> str:
    model = normalize_model_name(model)
    task = task.upper()
    task_type = TASK_TYPE[task]
    return os.path.join(
        results_root,
        model,
        f"cycle_{cycle}",
        task.lower(),
        "models",
        task_type,
    )


def checkpoint_path(results_root: str, model: str, task: str, cycle: int, fold_num: int) -> str:
    return os.path.join(
        _model_root(results_root, model, task, cycle),
        f"best_model_fold_{fold_num}.pth",
    )


def pretrain_path(results_root: str, model: str, task: str, cycle: int, fold_num: int) -> str:
    return os.path.join(
        _model_root(results_root, model, task, cycle),
        f"pretrain_model_fold_{fold_num}.pth",
    )


def run_save_path(results_root: str, model: str, task: str, cycle: int) -> str:
    """CrossValidator save_path (contains models/ and attention_weights/)."""
    model = normalize_model_name(model)
    return os.path.join(
        results_root,
        model,
        f"cycle_{cycle}",
        task.lower(),
    )


def prepare_fold_transfer(
    results_root: str,
    model: str,
    task: str,
    cycle: int,
    fold_num: int,
) -> Optional[str]:
    """
    Copy source best_model_fold_{fold_num}.pth → target pretrain_model_fold_{fold_num}.pth
    within the same architecture.
    """
    source = resolve_source(task, cycle)
    if source is None:
        print(f"→ No transfer source for {normalize_model_name(model)} {task} Cycle {cycle} (train from scratch)")
        return None

    src_task, src_cycle = source
    src = checkpoint_path(results_root, model, src_task, src_cycle, fold_num)
    dst = pretrain_path(results_root, model, task, cycle, fold_num)

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Transfer required for {normalize_model_name(model)} {task} Cycle {cycle} Fold {fold_num}, "
            f"but source checkpoint not found:\n  {src}\n"
            f"Expected source: {src_task} Cycle {src_cycle} Fold {fold_num}"
        )

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(
        f"→ Transfer prepared [{normalize_model_name(model)}]: "
        f"{src_task} C{src_cycle} fold {fold_num} → {task} C{cycle} fold {fold_num}"
    )
    print(f"  {src}")
    print(f"  → {dst}")
    return dst
