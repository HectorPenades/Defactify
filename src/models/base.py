"""Base abstract classes for models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseModel(ABC, nn.Module):
    """Abstract base class for models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.num_classes = config.get('model', {}).get('num_classes', 6)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze all parameters except classifier."""
        for name, param in self.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
