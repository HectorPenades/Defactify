"""Evaluation and inference pipeline."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from src.training.metrics import MetricsComputer


class Evaluator:
    """Evaluate model on dataset."""

    def __init__(self, model: nn.Module, device: str = 'cuda:0'):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict_batch(self, batch: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch. Handles single-input and fusion (two-input) models.

        Args:
            batch: Batch dictionary (must contain 'image'; may contain 'fft' for fusion)

        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            images = batch['image'].to(self.device)
            if 'fft' in batch:
                logits = self.model(images, batch['fft'].to(self.device))
            else:
                logits = self.model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

        return preds, probs

    def evaluate_on_split(self, dataloader: DataLoader, num_classes: int,
                         task: str = 'multiclass') -> Dict[str, Any]:
        """
        Evaluate on a complete split.

        Args:
            dataloader: Data loader
            num_classes: Number of classes
            task: Task type ('binary' or 'multiclass')

        Returns:
            Dictionary of evaluation results
        """
        all_preds = []
        all_probs = []
        all_labels = []
        all_image_ids = []

        pbar = tqdm(dataloader, desc="Evaluating", leave=True)
        for batch in pbar:
            preds, probs = self.predict_batch(batch)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch['label'].numpy())
            all_image_ids.extend(batch['image_id'])

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = MetricsComputer.compute_probabilities_metrics(
            all_probs, all_labels, num_classes, task
        )

        return {
            'metrics': metrics,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'image_ids': all_image_ids
        }

    def save_predictions(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save predictions to CSV.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save CSV
        """
        num_classes = results['probabilities'].shape[1]
        predictions_list = []

        for i, image_id in enumerate(results['image_ids']):
            pred_dict = {
                'image_id': image_id,
                'true_label': int(results['labels'][i]),
                'pred_label': int(results['predictions'][i])
            }

            # Add probability columns
            for class_idx in range(num_classes):
                pred_dict[f'prob_class_{class_idx}'] = float(
                    results['probabilities'][i, class_idx]
                )

            predictions_list.append(pred_dict)

        df = pd.DataFrame(predictions_list)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
