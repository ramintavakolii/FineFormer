import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_time_series_dataset(dataset_path, dataset_name, class_name):
    """
    General loader for both UCLA and COBRE time series data.

    Args:
        dataset_path (str): Path to parent data folder
        dataset_name (str): "UCLA" or "COBRE"
        class_name (str): Class folder name (e.g., "Healthy", "Bipolar", "Schizophrenia")

    Returns:
        np.array: time series data (n_subjects, timesteps, features)
    """

    print(f"Loading {dataset_name} {class_name} time series database ...")

    class_path = os.path.join(dataset_path, dataset_name, class_name)

    entries = os.listdir(class_path)
    folders = [f for f in entries if os.path.isdir(os.path.join(class_path, f))]
    folders_sorted = sorted(folders, key=lambda x: int(x[-5:]))
    print(folders_sorted)


    all_series = []

    for folder in folders_sorted:
        folder_path = os.path.join(class_path, folder)

        if not os.path.isdir(folder_path) or folder.startswith('.'):
            continue

        # Build filename depending on dataset type
        if dataset_name == "UCLA":
            subject_id = folder.split("-")[-1]
            csv_filename = f"sub-{subject_id}_time_series.csv"
        elif dataset_name == "COBRE":
            subject_id = folder[7:]  # e.g., 'ts_file_123' → '_123'
            csv_filename = f"Sub{subject_id}_time_series.csv"
        else:
            raise ValueError("Unsupported dataset name")

        csv_path = os.path.join(folder_path, csv_filename)

        if not os.path.exists(csv_path):
            print(f"Warning: Missing file for {folder}")
            continue

        # Read CSV and drop first column
        df = pd.read_csv(csv_path)
        arr = df.iloc[:, 1:].to_numpy()
        all_series.append(arr)

    if len(all_series) == 0:
        raise RuntimeError(f"No valid data found for {dataset_name} {class_name}")


    return np.array(all_series)


def load_and_merge_groups(dataset_path):
    """Notebook data aggregation (cell that builds X_*_all)."""
    ucla_healthy_X = load_time_series_dataset(dataset_path, "UCLA", "Healthy")
    ucla_schizo_X = load_time_series_dataset(dataset_path, "UCLA", "Schizophrenia")

    cobre_healthy_X = load_time_series_dataset(dataset_path, "COBRE", "Healthy")
    cobre_schizo_X = load_time_series_dataset(dataset_path, "COBRE", "Schizophrenia")

    ucla_bipolar_X = load_time_series_dataset(dataset_path, "UCLA", "Bipolar")

    X_healthy_all = np.vstack([cobre_healthy_X, ucla_healthy_X])  # (139, 142, 118)
    X_schizo_all = np.vstack([cobre_schizo_X, ucla_schizo_X])  # (120, 142, 118)
    X_bipolar_all = ucla_bipolar_X  # (49, 142, 118)

    print("Merged data shapes:")
    print(f"Healthy: {X_healthy_all.shape}")
    print(f"Schizo: {X_schizo_all.shape}")
    print(f"Bipolar: {X_bipolar_all.shape}")

    return X_healthy_all, X_schizo_all, X_bipolar_all


def create_all_fold_indices(X_healthy, X_schizo, X_bipolar, n_splits=5, random_state=35, save_path="./results/fold_indices/"):
    """
    Create KFold split indices for Healthy, Schizophrenia, and Bipolar groups.
    Stores both group-local and global indices (assuming merged order = [Healthy, Schizo, Bipolar]).

    Args:
        X_healthy: ndarray (n_healthy, timesteps, features)
        X_schizo: ndarray (n_schizo, timesteps, features)
        X_bipolar: ndarray (n_bipolar, timesteps, features)
        n_splits: number of folds
        random_state: seed for reproducibility
        save_path: directory to save indices

    Returns:
        dict: fold indices for all groups (train/val) with global indexing
    """

    print("Creating fold indices for all groups...")
    n_healthy, n_schizo, n_bipolar = len(X_healthy), len(X_schizo), len(X_bipolar)
    print(f"Healthy subjects: {n_healthy}")
    print(f"Schizophrenia subjects: {n_schizo}")
    print(f"Bipolar subjects: {n_bipolar}")

    # Create KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Local indices for each group
    healthy_idx = np.arange(n_healthy)
    schizo_idx = np.arange(n_schizo)
    bipolar_idx = np.arange(n_bipolar)

    # Offsets for global indices
    schizo_offset = n_healthy
    bipolar_offset = n_healthy + n_schizo

    # Generate folds per group
    healthy_folds = list(kf.split(healthy_idx))
    schizo_folds = list(kf.split(schizo_idx))
    bipolar_folds = list(kf.split(bipolar_idx))

    # Store everything
    all_fold_indices = {}
    for fold in range(n_splits):
        fold_dict = {
            # Healthy (global = same as local)
            "healthy_train_idx": healthy_folds[fold][0],
            "healthy_val_idx": healthy_folds[fold][1],

            # Schizophrenia (global = local + offset)
            "schizo_train_idx": schizo_folds[fold][0] + schizo_offset,
            "schizo_val_idx": schizo_folds[fold][1] + schizo_offset,

            # Bipolar (global = local + offset)
            "bipolar_train_idx": bipolar_folds[fold][0] + bipolar_offset,
            "bipolar_val_idx": bipolar_folds[fold][1] + bipolar_offset,
        }
        all_fold_indices[fold] = fold_dict

        # Print fold stats
        print(f"\nFold {fold+1}:")
        print(f"  Healthy - Train: {len(fold_dict['healthy_train_idx'])}, Val: {len(fold_dict['healthy_val_idx'])}")
        print(f"  Schizo  - Train: {len(fold_dict['schizo_train_idx'])}, Val: {len(fold_dict['schizo_val_idx'])}")
        print(f"  Bipolar - Train: {len(fold_dict['bipolar_train_idx'])}, Val: {len(fold_dict['bipolar_val_idx'])}")

    # Save fold indices
    os.makedirs(save_path, exist_ok=True)
    indices_path = os.path.join(save_path, "all_fold_indices.pkl")
    with open(indices_path, "wb") as f:
        pickle.dump(all_fold_indices, f)

    print(f"\n✓ Saved fold indices to: {indices_path}")
    return all_fold_indices
