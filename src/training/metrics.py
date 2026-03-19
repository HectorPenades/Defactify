"""Metrics computation for evaluation."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Any, Tuple


class MetricsComputer:
    """Compute evaluation metrics."""

    @staticmethod
    def compute_classification_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        task: str = 'multiclass'
    ) -> Dict[str, Any]:
        """
        Compute classification metrics.

        Args:
            predictions: Predicted class indices (N,)
            labels: Ground truth labels (N,)
            num_classes: Number of classes
            task: 'binary' or 'multiclass'

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'balanced_accuracy': balanced_accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }

        # Label_A (binary):     0=Real, 1=AI-Generated
        # Label_B (multiclass): 0=Real, 1=SD21, 2=SDXL, 3=SD3, 4=DALLE3, 5=Midjourney
        BINARY_CLASS_NAMES    = {0: 'real', 1: 'ai_generated'}
        MULTICLASS_CLASS_NAMES = {0: 'real', 1: 'sd21', 2: 'sdxl', 3: 'sd3', 4: 'dalle3', 5: 'midjourney'}
        names = BINARY_CLASS_NAMES if num_classes == 2 else MULTICLASS_CLASS_NAMES

        # Per-class metrics — always computed (essential for binary imbalance 1:5)
        per_class_metrics = {}
        for class_idx in range(num_classes):
            class_mask      = predictions == class_idx
            class_true_mask = labels == class_idx

            tp = int((class_mask & class_true_mask).sum())
            fp = int((class_mask & ~class_true_mask).sum())
            fn = int((~class_mask & class_true_mask).sum())

            prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_cls = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_metrics[names.get(class_idx, f'class_{class_idx}')] = {
                'precision': float(prec),
                'recall':    float(rec),
                'f1':        float(f1_cls),
                'support':   int(class_true_mask.sum()),
            }

        metrics['per_class'] = per_class_metrics
        return metrics

    @staticmethod
    def compute_probabilities_metrics(
        probabilities: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        task: str = 'multiclass'
    ) -> Dict[str, float]:
        """
        Compute metrics from probability outputs.

        Args:
            probabilities: Probability outputs (N, num_classes)
            labels: Ground truth labels (N,)
            num_classes: Number of classes
            task: 'binary' or 'multiclass'

        Returns:
            Dictionary of metrics
        """
        predictions = np.argmax(probabilities, axis=1)

        metrics = MetricsComputer.compute_classification_metrics(
            predictions, labels, num_classes, task
        )

        # Add ROC-AUC for binary classification
        if num_classes == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(labels, probabilities[:, 1])
            except:
                metrics['roc_auc'] = 0.0

        return metrics
