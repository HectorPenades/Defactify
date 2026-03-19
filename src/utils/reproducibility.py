"""Reproducibility utilities for deterministic training."""

import numpy as np
import torch
import random
from typing import Set


class ReproducibilityManager:
    """Manager for ensuring reproducible experiments."""

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set all random seeds for reproducibility.

        Args:
            seed: Random seed value
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def verify_no_data_leakage(
        train_ids: Set[str],
        val_ids: Set[str],
        test_ids: Set[str]
    ) -> bool:
        """
        Verify that there is no overlap between train, val, and test sets.

        Args:
            train_ids: Set of training image IDs
            val_ids: Set of validation image IDs
            test_ids: Set of test image IDs

        Returns:
            True if no leakage, raises AssertionError otherwise
        """
        train_ids = set(train_ids)
        val_ids = set(val_ids)
        test_ids = set(test_ids)

        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids

        assert len(train_val_overlap) == 0, \
            f"Data leakage: {len(train_val_overlap)} images in both train and val"
        assert len(train_test_overlap) == 0, \
            f"Data leakage: {len(train_test_overlap)} images in both train and test"
        assert len(val_test_overlap) == 0, \
            f"Data leakage: {len(val_test_overlap)} images in both val and test"

        return True
