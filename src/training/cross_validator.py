import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR

from src.data.dataset import FMriDatasetAug
from src.training.trainer import Trainer
from src.training.attention_extractor import AttentionExtractor

class CrossValidator:
    def __init__(self, X_healthy, X_schizo, X_bipolar, device, model_class, model_params, scheduler_factor=0.5 ,fold_indices_path=None,
                 n_splits=5, batch_size=16, num_epochs=50, patience=7, random_state=41, initial_lr=3e-4, fine_tune=True,
                 reinit_classifier=True, freeze_encoder=False, weight_decay=1e-2, class_names=None, save_path="./results/"):

        # Data groups
        self.X_healthy= X_healthy
        self.X_schizo = X_schizo
        self.X_bipolar = X_bipolar

        # Training setup
        self.scheduler_factor, self.device = scheduler_factor, device
        self.model_class, self.model_params = model_class, model_params
        self.n_folds, self.batch_size = n_splits, batch_size
        self.num_epochs, self.patience = num_epochs, patience
        self.random_state, self.initial_lr = random_state, initial_lr
        self.fine_tune, self.reinit_classifier = fine_tune, reinit_classifier
        self.freeze_encoder, self.weight_decay = freeze_encoder, weight_decay
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.save_path = save_path

        # Indices
        self.fold_indices_path = fold_indices_path
        self.fold_indices = self._load_fold_indices()

        # Results
        self.all_fold_metrics, self.all_training_histories = [], []
        self.metrics_summary = {}

        # Save dirs
        os.makedirs(self.save_path, exist_ok=True)
        self.model_save_path = os.path.join(self.save_path, "models")
        self.attention_save_path = os.path.join(self.save_path, "attention_weights")
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.attention_save_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Data Handling
    # ------------------------------------------------------------------
    def _load_fold_indices(self):
        if not (self.fold_indices_path and os.path.exists(self.fold_indices_path)):
            raise FileNotFoundError(f"Fold indices not found: {self.fold_indices_path}")
        with open(self.fold_indices_path, "rb") as f:
            print(f"✓ Loaded fold indices from {self.fold_indices_path}")
            return pickle.load(f)

    def _prepare_split(self, X_a, idx_a, label_a,
                    X_b, idx_b, label_b):
        """Helper: stack two groups and assign constant labels"""
        X_a_sel = X_a[idx_a]
        X_b_sel = X_b[idx_b]

        X = np.vstack([X_a_sel, X_b_sel])

        y_a = np.full((X_a_sel.shape[0], 1), label_a, dtype=int)
        y_b = np.full((X_b_sel.shape[0], 1), label_b, dtype=int)

        y = np.vstack([y_a, y_b])
        return X, y

    def _print_split_stats(self, task, fold, X_train, y_train, X_val, y_val):
        print(f"\nTask: {task}, Fold: {fold+1}")
        print(f"→ Train: {len(X_train)} samples | Val: {len(X_val)} samples")
        print(f"→ Train dist: {np.bincount(y_train.flatten())}")
        print(f"→ Val dist: {np.bincount(y_val.flatten())}")

    def _build_dataloaders(self, X_train, y_train, X_val, y_val):

            train_ds = FMriDatasetAug(X_train, y_train, augment=True)
            val_ds = FMriDatasetAug(X_val, y_val, augment=False)

            return (DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
                    DataLoader(val_ds, batch_size=self.batch_size, shuffle=False))

    def create_task_dataloaders(self, task_type, fold_num):
        idx = self.fold_indices[fold_num]

        if task_type == "healthy_vs_schizo":
            # Convert global to local indices
            healthy_train_local = idx['healthy_train_idx']  # Already local (0-138)
            healthy_val_local = idx['healthy_val_idx']      # Already local (0-138)
            schizo_train_local = idx['schizo_train_idx'] - len(self.X_healthy)  # Convert: global → local
            schizo_val_local = idx['schizo_val_idx'] - len(self.X_healthy)      # Convert: global → local

            X_train, y_train = self._prepare_split(self.X_healthy, healthy_train_local, 1,
                                                   self.X_schizo, schizo_train_local, 0)
            X_val, y_val = self._prepare_split(self.X_healthy, healthy_val_local, 1,
                                               self.X_schizo, schizo_val_local, 0)

        elif task_type == "healthy_vs_bipolar":
            # Convert global to local indices
            healthy_train_local = idx['healthy_train_idx']                                               # Already local (0-138)
            healthy_val_local = idx['healthy_val_idx']                                                   # Already local (0-138)
            bipolar_train_local = idx['bipolar_train_idx'] - (len(self.X_healthy) + len(self.X_schizo))  # Convert: global → local
            bipolar_val_local = idx['bipolar_val_idx'] - (len(self.X_healthy) + len(self.X_schizo))      # Convert: global → local

            X_train, y_train = self._prepare_split(self.X_healthy, healthy_train_local, 1,
                                                   self.X_bipolar, bipolar_train_local, 0)
            X_val, y_val = self._prepare_split(self.X_healthy, healthy_val_local, 1,
                                               self.X_bipolar, bipolar_val_local, 0)

        elif task_type == "bipolar_vs_schizo":
            # Convert global to local indices
            bipolar_train_local = idx['bipolar_train_idx'] - (len(self.X_healthy) + len(self.X_schizo))  # Convert: global → local
            bipolar_val_local = idx['bipolar_val_idx'] - (len(self.X_healthy) + len(self.X_schizo))      # Convert: global → local
            schizo_train_local = idx['schizo_train_idx'] - len(self.X_healthy)                           # Convert: global → local
            schizo_val_local = idx['schizo_val_idx'] - len(self.X_healthy)                               # Convert: global → local

            X_train, y_train = self._prepare_split(self.X_bipolar, bipolar_train_local, 1,
                                                   self.X_schizo, schizo_train_local, 0)
            X_val, y_val = self._prepare_split(self.X_bipolar, bipolar_val_local, 1,
                                               self.X_schizo, schizo_val_local, 0)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        self._print_split_stats(task_type, fold_num, X_train, y_train, X_val, y_val)
        return self._build_dataloaders(X_train, y_train, X_val, y_val)


    # ------------------------------------------------------------------
    # Training & Evaluation
    # ------------------------------------------------------------------
    def train_fold(self, fold_num, train_loader, val_loader, task_name="healthy_vs_schizo"):
        model = self.model_class(**self.model_params).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=self.scheduler_factor, patience=10)

        trainer = Trainer(model, train_loader, val_loader, self.device,
                         criterion, optimizer, scheduler, fold_num,
                         save_path=os.path.join(self.model_save_path, task_name),
                         num_epochs=self.num_epochs, patience=self.patience,
                         initial_lr=self.initial_lr,
                         fine_tune=self.fine_tune,
                         reinit_classifier=self.reinit_classifier,
                         freeze_encoder=self.freeze_encoder)
        return trainer.run()

    def extract_fold_attention(self, best_model, val_loader, fold_num, task_name):
        extractor = AttentionExtractor(best_model, val_loader, self.device, fold_num,
                                       save_dir=os.path.join(self.attention_save_path, task_name), average_heads=True)
        extractor.run()

    def run_task(self, task_type="healthy_vs_schizo", fold_indices=None):
        print(f"\n{'='*80}\nRUNNING TASK: {task_type.upper()}\n{'='*80}")

        if fold_indices is None:
            fold_indices = list(range(self.n_folds))

        for fold in fold_indices:
            print(f"\n{'='*60}\nFOLD {fold+1}/{self.n_folds} - {task_type}\n{'='*60}")
            train_loader, val_loader = self.create_task_dataloaders(task_type, fold)

            best_model, metrics, history = self.train_fold(fold+1, train_loader, val_loader, task_type)
            self.all_fold_metrics.append(metrics)
            self.all_training_histories.append(history)

            try:
                self.extract_fold_attention(best_model, val_loader, fold+1, task_type)
                print("Attention weights saved")
            except Exception as e:
                print(f"Warning: Attention extraction failed → {e}")

            del best_model
            torch.cuda.empty_cache()

        self.aggregate_cv_results()
        self.save_cv_summary(task_type)
        return self.all_fold_metrics, self.metrics_summary

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def aggregate_cv_results(self):
        print("\n" + "="*60 + "\nCROSS-VALIDATION SUMMARY\n" + "="*60)
        self.metrics_summary = {}

        for m in ['accuracy', 'balanced_accuracy', 'f1', 'precision',
                  'recall', 'sensitivity', 'auc', 'specificity', 'npv', 'ppv']:
            if all(m in r for r in self.all_fold_metrics):
                vals = [r[m] for r in self.all_fold_metrics]
                self.metrics_summary[m] = {'mean': np.mean(vals),
                                          'std': np.std(vals),
                                          'scores': vals}
                print(f"{m:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    def save_cv_summary(self, task_name):
        path = os.path.join(self.save_path, f"cv_summary_{task_name}.pt")
        torch.save({'all_fold_metrics': self.all_fold_metrics,
                   'metrics_summary': self.metrics_summary,
                   'training_histories': self.all_training_histories,
                   'task_name': task_name,
                   'config': {
                       'n_splits': self.n_folds,
                       'batch_size': self.batch_size,
                       'num_epochs': self.num_epochs,
                       'patience': self.patience,
                       'initial_lr': self.initial_lr,
                       'weight_decay': self.weight_decay,
                       'class_names': self.class_names,
                       'model_class': self.model_class.__name__,
                       'model_params': self.model_params}},
                   path)
        print(f"\n✓ Saved CV summary → {path}")
        print(f"✓ Models → {os.path.join(self.model_save_path, task_name)}")
        print(f"✓ Attention weights → {os.path.join(self.attention_save_path, task_name)}")