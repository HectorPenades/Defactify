"""Defactify dataset implementation."""

import torch
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
from datasets import load_dataset as hf_load_dataset

from src.data.base import BaseDataset
from src.data.transforms import ImagePreprocessor
from src.processing.fft import compute_fft_grayscale, compute_fft_perchannel
from src.utils.cache_manager import FFTCacheManager


# Dataset structure (Rajarshi-Roy-research/Defactify_Image_Dataset):
#   Columns: Caption (str), Image (PIL), Label_A (int), Label_B (int)
#   Label_A — binary:     0=Real, 1=AI-Generated
#   Label_B — multiclass: 0=Real, 1=SD21, 2=SDXL, 3=SD3, 4=DALLE3, 5=Midjourney

LABEL_B_TO_SOURCE = {
    0: 'real',
    1: 'sd21',
    2: 'sdxl',
    3: 'sd3',
    4: 'dalle',
    5: 'midjourney',
}


class DefactifyDataset(BaseDataset):
    """Defactify dataset for AI image detection.

    Supported modes (config.dataset.mode):
      - 'rgb'            : Standard RGB images  → batch['image'] (3,H,W)
      - 'fft_grayscale'  : Grayscale FFT        → batch['image'] (1,H,W)
      - 'fft_perchannel' : Per-channel FFT       → batch['image'] (3,H,W)
      - 'fusion_late'    : RGB + FFT             → batch['image'] (3,H,W) + batch['fft'] (C,H,W)
    """

    CLASS_MAPPING = {v: k for k, v in LABEL_B_TO_SOURCE.items()}   # name → int
    REVERSE_CLASS_MAPPING = LABEL_B_TO_SOURCE                        # int → name

    def __init__(self, config: Dict[str, Any], split: str = 'train',
                 transform: Optional[Any] = None):
        """
        Args:
            config: Configuration dictionary
            split:  'train' | 'validation' | 'test'
            transform: Optional torchvision transforms (applied to RGB channel)
        """
        super().__init__(config, split, transform)

        self.mode = config.get('dataset', {}).get('mode', 'rgb')
        self.task = config.get('task', 'multiclass')
        self.hf_repo = config.get('dataset', {}).get(
            'hf_repo', 'Rajarshi-Roy-research/Defactify_Image_Dataset'
        )
        self.preprocessor = ImagePreprocessor(config)

        # FFT cache (used for fft_* and fusion_late modes)
        cache_dir = config.get('cache', {}).get('cache_dir', './data/cache')
        fft_mode = self.mode if self.mode != 'fusion_late' else (
            'fft_grayscale'
            if config.get('dataset', {}).get('fft_channels', 1) == 1
            else 'fft_perchannel'
        )
        self._fft_cache = FFTCacheManager(cache_dir, fft_mode)
        self._fft_mode = fft_mode

        self._load_hf_dataset()
        self._build_indices()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_hf_dataset(self) -> None:
        """Load the requested split directly from HuggingFace."""
        try:
            self.hf_dataset = hf_load_dataset(
                self.hf_repo,
                split=self.split,
                keep_in_memory=False
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load split '{self.split}' from {self.hf_repo}: {e}"
            )

    def _build_indices(self) -> None:
        """One HF row = one sample."""
        if self.split not in ('train', 'validation', 'test'):
            raise ValueError(
                f"Unknown split: {self.split}. "
                "Must be one of ['train', 'validation', 'test']"
            )
        self.num_samples = len(self.hf_dataset)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _to_pil(self, raw) -> Image.Image:
        """Convert HF image field to PIL RGB."""
        if isinstance(raw, Image.Image):
            return raw.convert('RGB')
        if isinstance(raw, np.ndarray):
            return Image.fromarray(raw).convert('RGB')
        if hasattr(raw, 'convert'):
            return raw.convert('RGB')
        return raw

    def _get_fft_tensor(self, image: Image.Image, image_id: str) -> torch.Tensor:
        """
        Return cached FFT tensor, computing it on first access.

        Returns:
            (1,H,W) for fft_grayscale, (3,H,W) for fft_perchannel
        """
        if self._fft_cache.exists(image_id, self.split):
            arr = self._fft_cache.load(image_id, self.split)
        else:
            if self._fft_mode == 'fft_grayscale':
                arr = compute_fft_grayscale(image)    # (H,W)
            else:
                arr = compute_fft_perchannel(image)   # (H,W,3)
            self._fft_cache.save(image_id, self.split, arr)

        if arr.ndim == 2:
            return torch.from_numpy(arr).unsqueeze(0)   # (1,H,W)
        else:
            return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict with keys:
            'image'    : main input tensor
            'label'    : long tensor
            'source'   : str class name ('real', 'sd21', ...)
            'image_id' : str
            'caption'  : str (may be empty)
        For fusion_late also includes:
            'fft'      : FFT tensor (C,H,W)
        """
        example = self.hf_dataset[idx]

        image = self._to_pil(example['Image'])
        image = self.preprocessor.resize_image(image)

        label_a = int(example['Label_A'])   # binary:     0=real, 1=AI
        label_b = int(example['Label_B'])   # multiclass: 0-5
        source = LABEL_B_TO_SOURCE.get(label_b, 'unknown')
        image_id = str(idx)

        # ---- Build output based on mode ----
        if self.mode == 'rgb':
            transform = self.transform or self.preprocessor.get_test_transforms()
            image_tensor = transform(image)
            item = {'image': image_tensor}

        elif self.mode == 'fft_grayscale':
            item = {'image': self._get_fft_tensor(image, image_id)}

        elif self.mode == 'fft_perchannel':
            item = {'image': self._get_fft_tensor(image, image_id)}

        elif self.mode == 'fusion_late':
            transform = self.transform or self.preprocessor.get_test_transforms()
            item = {
                'image': transform(image),
                'fft': self._get_fft_tensor(image, image_id),
            }

        else:
            raise ValueError(
                f"Unknown dataset mode: {self.mode}. "
                "Must be one of: rgb, fft_grayscale, fft_perchannel, fusion_late"
            )

        # Label selection
        label = label_a if self.task == 'binary' else label_b

        item.update({
            'label': torch.tensor(label, dtype=torch.long),
            'source': source,
            'image_id': image_id,
            'caption': example.get('Caption', ''),
        })
        return item

    # ------------------------------------------------------------------

    def _validate_no_leakage(self) -> bool:
        """HF native splits guarantee no leakage."""
        return True

    @property
    def num_classes(self) -> int:
        return 2 if self.task == 'binary' else 6
