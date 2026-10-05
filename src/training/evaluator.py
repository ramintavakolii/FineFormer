
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)


class ModelEvaluator:
    def __init__(self, model, device, criterion=None):
        """
        Binary classification evaluator.

        Label in this project:
        - Label 0: Positive class  (SC, BD depending on task)
        - Label 1: Negative class  (HC or non-positive group)

        healthy_vs_schizo: 1 , 0
        healthy_vs_bipolar: 1, 0
        bipolar_vs_schizo: 1, 0

        Model output convention:
        - If output dim == 1 → sigmoid probability for positive class (label 0)
        - If output dim == 2 → softmax probabilities [P(label0), P(label1)]
        """
        self.model = model
        self.device = device
        self.criterion = criterion

        self.all_preds = None
        self.all_labels = None
        self.all_probs = None
        self.all_losses = None
        self.metrics = None

    def collect_predictions(self, dataloader):
        self.model.eval()
        preds, labels, probs, losses = [], [], [], []

        with torch.no_grad():
            for inputs, label in dataloader:
                inputs, label = inputs.to(self.device), label.to(self.device).view(-1)
                outputs = self.model(inputs)

                # Calculate loss if criterion provided
                if self.criterion is not None:
                    loss = self.criterion(outputs, label)   # label shape (batch_size, 1) and outputs shape (batch_size, 2)
                    losses.extend([loss.item()] * inputs.size(0))  # Per sample loss

                if outputs.shape[1] == 1:
                    # Sigmoid for probability of label 0 (positive class)
                    prob_pos = torch.sigmoid(outputs).squeeze(1)
                    pred = (prob_pos >= 0.5).long()  # 1 if prob >= 0.5 else 0
                else:
                    # Softmax: column 0 = probability of label 0 (positive class)
                    prob_pos = torch.softmax(outputs, dim=1)[:, 0]
                    pred = torch.argmax(outputs, dim=1)

                preds.extend(pred.cpu().numpy())
                labels.extend(label.cpu().numpy())
                probs.extend(prob_pos.cpu().numpy())

        self.all_preds = np.array(preds)
        self.all_labels = np.array(labels)
        self.all_probs = np.array(probs)  # Probability of label 0 (positive class)
        self.all_losses = np.array(losses) if losses else None

    def calculate_binary_diagnostics(self, cm):
        """
        Sensitivity, Specificity, PPV, NPV for positive class = label 0.

        Confusion matrix layout for labels [0, 1]:
        cm[actual, predicted]
            [[TP, FN],   # Actual 0: predicted as 0 (TP), predicted as 1 (FN)
             [FP, TN]]   # Actual 1: predicted as 0 (FP), predicted as 1 (TN)
        """
        tp = cm[0, 0]  # True Positives: actual 0, predicted 0
        fn = cm[0, 1]  # False Negatives: actual 0, predicted 1
        fp = cm[1, 0]  # False Positives: actual 1, predicted 0
        tn = cm[1, 1]  # True Negatives: actual 1, predicted 1

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall for positive class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Recall for negative class
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision for positive class
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Precision for negative class

        return sensitivity, specificity, ppv, npv

    def calculate_metrics(self):
        # Basic metrics for positive class (label=0)
        acc = accuracy_score(self.all_labels, self.all_preds)
        bal_acc = balanced_accuracy_score(self.all_labels, self.all_preds)
        f1 = f1_score(self.all_labels, self.all_preds, pos_label=0)
        precision_pos = precision_score(self.all_labels, self.all_preds, pos_label=0, zero_division=0)
        recall_pos = recall_score(self.all_labels, self.all_preds, pos_label=0, zero_division=0)

        # FIXED: AUC calculation - use probabilities for negative class (label 1)
        # Since roc_auc_score expects probabilities for the positive class (which is label 1 in sklearn's convention)
        auc_score = roc_auc_score(self.all_labels, 1 - self.all_probs)

        # Confusion matrix & diagnostic metrics
        cm = confusion_matrix(self.all_labels, self.all_preds, labels=[0, 1])
        sensitivity, specificity, ppv, npv = self.calculate_binary_diagnostics(cm)

        avg_val_loss = np.mean(self.all_losses) if self.all_losses is not None else None

        self.metrics = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "precision": precision_pos,  # Same as PPV
            "recall": recall_pos,        # Same as Sensitivity for label 0
            "auc": auc_score,
            "sensitivity": sensitivity,  # Same as recall for positive class
            "specificity": specificity,
            "ppv": ppv,                 # Same as precision for positive class
            "npv": npv,
            "confusion_matrix": cm,
            "val_loss": avg_val_loss
        }

    def evaluate(self, dataloader):
        self.collect_predictions(dataloader)
        self.calculate_metrics()
        return self.metrics

    def summarize_metrics(self, verbose=True):
        if self.metrics is None:
            raise ValueError("No metrics found. Run evaluate() first.")

        summary_str = "\n" + "="*50 + "\nBINARY CLASSIFICATION METRICS\n" + "="*50
        summary_str += f"\nAccuracy: {self.metrics['accuracy']:.4f}"
        summary_str += f"\nBalanced Accuracy: {self.metrics['balanced_accuracy']:.4f}"
        summary_str += f"\nF1 Score (pos=0): {self.metrics['f1']:.4f}"
        summary_str += f"\nPrecision/PPV (pos=0): {self.metrics['precision']:.4f}"
        summary_str += f"\nSensitivity (pos=0): {self.metrics['sensitivity']:.4f}"
        summary_str += f"\nRecall (pos=1): {self.metrics['recall']:.4f}"
        summary_str += f"\nSpecificity: {self.metrics['specificity']:.4f}"
        summary_str += f"\nNPV: {self.metrics['npv']:.4f}"
        summary_str += f"\nPPV: {self.metrics['ppv']:.4f}"
        summary_str += f"\nAUC: {self.metrics['auc']:.4f}"
        if self.metrics['val_loss'] is not None:
            summary_str += f"\nValidation Loss: {self.metrics['val_loss']:.4f}"
        summary_str += f"\nConfusion Matrix (rows=actual, cols=predicted):\n{self.metrics['confusion_matrix']}"
        summary_str += f"\n  [[TP={self.metrics['confusion_matrix'][0,0]}, FN={self.metrics['confusion_matrix'][0,1]}],"
        summary_str += f"\n   [FP={self.metrics['confusion_matrix'][1,0]}, TN={self.metrics['confusion_matrix'][1,1]}]]"

        if verbose:
            print(summary_str)
        else:
            return summary_str
