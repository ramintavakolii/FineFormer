import os
import pickle
import shutil

import numpy as np

from src.data.preprocessing import (
    load_and_merge_groups,
    create_all_fold_indices,
)


def make_synthetic_groups(n_healthy=139, n_schizo=120, n_bipolar=49,
                          n_timesteps=142, n_rois=118, seed=0):
    """Synthetic stand-in for explicit smoke / parity tests (--synthetic only)."""
    rng = np.random.RandomState(seed)
    X_healthy = rng.randn(n_healthy, n_timesteps, n_rois).astype(np.float32)
    X_schizo = rng.randn(n_schizo, n_timesteps, n_rois).astype(np.float32)
    X_bipolar = rng.randn(n_bipolar, n_timesteps, n_rois).astype(np.float32)
    return X_healthy, X_schizo, X_bipolar


class DataModule:
    def __init__(self, dataset_path, fold_indices_path, fold_random_state=40,
                 n_splits=5, use_synthetic=False):
        self.dataset_path = dataset_path
        self.fold_indices_path = fold_indices_path
        self.fold_random_state = fold_random_state
        self.n_splits = n_splits
        self.use_synthetic = bool(use_synthetic)

        self.X_healthy = None
        self.X_schizo = None
        self.X_bipolar = None

    def setup(self):
        if self.use_synthetic:
            print("→ Synthetic data mode (--synthetic / data.use_synthetic=true)")
            print("  This is smoke-test data only — not a real-data experiment.")
            self.X_healthy, self.X_schizo, self.X_bipolar = make_synthetic_groups()
        else:
            path = self.dataset_path
            if not path or not os.path.isdir(path):
                raise FileNotFoundError(
                    "Real fMRI dataset was not found at:\n"
                    f"  {path!r}\n\n"
                    "Set FMRI_DATA_PATH to the real dataset location, or set "
                    "data.dataset_path in configs/config.yaml.\n"
                    "For synthetic smoke testing, explicitly pass --synthetic.\n"
                    "Missing data must never be treated as a successful experiment."
                )
            self.X_healthy, self.X_schizo, self.X_bipolar = load_and_merge_groups(path)
            if (
                len(self.X_healthy) == 0
                or len(self.X_schizo) == 0
                or len(self.X_bipolar) == 0
            ):
                raise RuntimeError(
                    f"Loaded empty group(s) from {path!r}: "
                    f"H={len(self.X_healthy)}, S={len(self.X_schizo)}, "
                    f"B={len(self.X_bipolar)}"
                )

        self._ensure_fold_indices()
        return self

    def _ensure_fold_indices(self):
        if os.path.exists(self.fold_indices_path):
            return
        save_dir = os.path.dirname(self.fold_indices_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        all_fold_indices = create_all_fold_indices(
            self.X_healthy, self.X_schizo, self.X_bipolar,
            n_splits=self.n_splits,
            random_state=self.fold_random_state,
            save_path=save_dir,
        )
        # create_all_fold_indices always writes all_fold_indices.pkl inside save_path
        produced = os.path.join(save_dir, "all_fold_indices.pkl")
        if os.path.abspath(produced) != os.path.abspath(self.fold_indices_path):
            if not os.path.exists(produced):
                raise FileNotFoundError(
                    f"Expected fold indices at {produced!r} after creation, but file is missing."
                )
            os.makedirs(os.path.dirname(self.fold_indices_path) or ".", exist_ok=True)
            shutil.copy2(produced, self.fold_indices_path)

        if not os.path.exists(self.fold_indices_path):
            raise FileNotFoundError(
                f"Fold indices were not created at {self.fold_indices_path!r}"
            )
