import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False):
    if seed is None:
        print("→ Seeding: disabled (historical-like mode)")
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"⚠ deterministic algorithms unavailable: {e}")

    print(f"→ Seeding: enabled (seed={seed}, deterministic={deterministic})")
