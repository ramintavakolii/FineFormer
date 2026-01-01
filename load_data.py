import os
import numpy as np
import pandas as pd

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


def global_roi_standardize_all(X):
    """
    Standardizes the entire dataset globally (across all subjects and time points).

    X: np.array of shape (n_subjects, time_steps, n_features)

    Returns:
        X_norm: normalized array
        mean: mean per ROI
        std: std per ROI
    """
    (139, 142, 118)
    # Compute mean and std for each ROI feature across all subjects & time points
    mean = X.reshape(-1, X.shape[2]).mean(axis=0)
    std = X.reshape(-1, X.shape[2]).std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1e-8

    # Normalize the whole dataset
    X_norm = (X - mean) / std

    return X_norm, mean, std