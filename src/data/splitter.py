"""Dataset splitting utilities for stratified train/val/test splits."""

import numpy as np
from pathlib import Path
from typing import Tuple
import pickle


class DatasetSplitter:
    """Creates reproducible stratified splits from a single dataset."""

    def __init__(
        self,
        dataset_size: int,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        cache_dir: Path = None
    ):
        """
        Initialize splitter for train/validation/test split.

        Args:
            dataset_size: Total number of samples in dataset
            split_ratios: Tuple of (train_ratio, val_ratio, test_ratio)
            seed: Random seed for reproducibility
            cache_dir: Directory to cache splits
        """
        self.dataset_size = dataset_size
        self.split_ratios = split_ratios
        self.seed = seed
        self.cache_dir = cache_dir

        # Validate ratios sum to 1.0
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

        # Create splits
        rng = np.random.RandomState(seed)
        indices = np.arange(dataset_size)
        rng.shuffle(indices)

        # Calculate split points
        train_size = int(dataset_size * split_ratios[0])
        val_size = int(dataset_size * split_ratios[1])

        self.train_indices = sorted(indices[:train_size].tolist())
        self.val_indices = sorted(indices[train_size : train_size + val_size].tolist())
        self.test_indices = sorted(indices[train_size + val_size :].tolist())

    def get_split(self, split: str) -> np.ndarray:
        """
        Get indices for a specific split.

        Args:
            split: 'train', 'validation', or 'test'

        Returns:
            Sorted array of indices
        """
        if split == 'train':
            return np.array(self.train_indices)
        elif split == 'validation':
            return np.array(self.val_indices)
        elif split == 'test':
            return np.array(self.test_indices)
        else:
            raise ValueError(f"Unknown split: {split}")

    def save_splits(self, cache_dir: Path) -> None:
        """Save splits to cache directory."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache_dir / 'train_indices.pkl', 'wb') as f:
            pickle.dump(self.train_indices, f)
        with open(cache_dir / 'val_indices.pkl', 'wb') as f:
            pickle.dump(self.val_indices, f)
        with open(cache_dir / 'test_indices.pkl', 'wb') as f:
            pickle.dump(self.test_indices, f)

    @staticmethod
    def load_splits(cache_dir: Path) -> Tuple[list, list, list]:
        """
        Load splits from cache directory.

        Args:
            cache_dir: Directory containing train_indices.pkl, etc.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        cache_dir = Path(cache_dir)

        with open(cache_dir / 'train_indices.pkl', 'rb') as f:
            train_indices = pickle.load(f)
        with open(cache_dir / 'val_indices.pkl', 'rb') as f:
            val_indices = pickle.load(f)
        with open(cache_dir / 'test_indices.pkl', 'rb') as f:
            test_indices = pickle.load(f)

        return train_indices, val_indices, test_indices
