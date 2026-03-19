"""Loss functions for training."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with class weights for imbalanced datasets."""

    def __init__(self, weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        """
        Initialize weighted cross-entropy loss.

        Args:
            weights: Tensor of class weights
            reduction: Reduction method ('mean', 'sum')
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weights, reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.

        Args:
            logits: Model output logits (B, num_classes)
            labels: Ground truth labels (B,)

        Returns:
            Loss value
        """
        return self.criterion(logits, labels)


def get_loss_function(config: Dict[str, Any],
                      class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Get loss function based on configuration.

    Args:
        config: Configuration dictionary
        class_weights: Optional class weights for imbalanced data

    Returns:
        Loss function module
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'crossentropy')

    if loss_type == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(weights=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
