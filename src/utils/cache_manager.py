"""Disk cache manager for precomputed FFT tensors."""

import numpy as np
from pathlib import Path
from typing import Optional


class FFTCacheManager:
    """
    Save and load FFT arrays to/from disk using .npy format.

    Cache layout:
        <cache_dir>/<mode>/<split>/<image_id>.npy

    Usage:
        cache = FFTCacheManager('./data/cache/fft', mode='fft_grayscale')
        if cache.exists(image_id, split):
            arr = cache.load(image_id, split)
        else:
            arr = compute_fft(...)
            cache.save(image_id, split, arr)
    """

    def __init__(self, cache_dir: str, mode: str):
        """
        Args:
            cache_dir: Root cache directory path
            mode: FFT mode string used as sub-folder ('fft_grayscale' | 'fft_perchannel')
        """
        self.root = Path(cache_dir) / mode
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, image_id: str, split: str) -> Path:
        split_dir = self.root / split
        split_dir.mkdir(exist_ok=True)
        safe_id = str(image_id).replace('/', '_').replace(' ', '_')
        return split_dir / f"{safe_id}.npy"

    def exists(self, image_id: str, split: str) -> bool:
        """Return True if cached file exists."""
        return self._path(image_id, split).exists()

    def save(self, image_id: str, split: str, data: np.ndarray) -> None:
        """Save numpy array to cache."""
        np.save(self._path(image_id, split), data)

    def load(self, image_id: str, split: str) -> np.ndarray:
        """Load numpy array from cache."""
        return np.load(self._path(image_id, split))
