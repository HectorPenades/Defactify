"""Base abstract classes for datasets."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for datasets."""

    def __init__(self, config: Dict[str, Any], split: str = 'train',
                 transform: Optional[Any] = None):
        """
        Initialize dataset.

        Args:
            config: Configuration dictionary
            split: Dataset split (train, validation, test)
            transform: Transform to apply to images
        """
        self.config = config
        self.split = split
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item by index.

        Returns:
            Dictionary with 'image', 'label', and additional keys
        """
        pass

    def _validate_no_leakage(self) -> bool:
        """Verify no data leakage between splits."""
        raise NotImplementedError
