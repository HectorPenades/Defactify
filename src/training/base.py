"""Base abstract classes for trainers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Abstract base class for trainers."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.model.to(self.device)

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        pass

    @abstractmethod
    def train(self, num_epochs: int, patience: int = 10) -> None:
        """Full training loop."""
        pass

    def save_checkpoint(self, path: str, epoch: int,
                       metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
