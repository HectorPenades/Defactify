"""Image transforms and preprocessing."""

import cv2
import numpy as np
from typing import Tuple
from PIL import Image
import torchvision.transforms as transforms


class ImagePreprocessor:
    """Handles image preprocessing and augmentation."""

    def __init__(self, config: dict):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.image_size = config.get('preprocessing', {}).get('image_size', 224)
        self.normalize_mean = config.get('preprocessing', {}).get(
            'normalize_mean', [0.485, 0.456, 0.406]
        )
        self.normalize_std = config.get('preprocessing', {}).get(
            'normalize_std', [0.229, 0.224, 0.225]
        )
        self.resize_method = config.get('preprocessing', {}).get(
            'resize_method', 'aspect_ratio_pad'
        )

    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image preserving aspect ratio with padding.

        Args:
            image: PIL Image

        Returns:
            Resized PIL Image
        """
        if self.resize_method == 'aspect_ratio_pad':
            return self._resize_aspect_ratio_pad(image)
        else:
            return self._resize_direct(image)

    def _resize_aspect_ratio_pad(self, image: Image.Image) -> Image.Image:
        """Resize preserving aspect ratio and pad to target size."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Calculate scaling factor
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        img_resized = cv2.resize(img_array, (new_w, new_h))

        # Pad to target size
        top = (self.image_size - new_h) // 2
        bottom = self.image_size - new_h - top
        left = (self.image_size - new_w) // 2
        right = self.image_size - new_w - left

        is_rgb = len(img_resized.shape) == 3
        pad_value = (0, 0, 0) if is_rgb else 0

        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_value
        )

        return Image.fromarray(img_padded)

    def _resize_direct(self, image: Image.Image) -> Image.Image:
        """Direct resize without preserving aspect ratio."""
        return image.resize((self.image_size, self.image_size), Image.BILINEAR)

    def get_train_transforms(self) -> transforms.Compose:
        """Get training transforms with augmentation."""
        aug_config = self.config.get('augmentations', {})

        transform_list = [
            transforms.ToTensor(),
        ]

        if aug_config.get('enabled', True):
            # Insert augmentation before normalization
            transform_list.insert(0, transforms.RandomHorizontalFlip(
                p=aug_config.get('random_flip', 0.5)
            ))
            if aug_config.get('color_jitter', 0) > 0:
                transform_list.insert(1, transforms.ColorJitter(
                    brightness=aug_config.get('brightness', 0.1),
                    contrast=aug_config.get('contrast', 0.1),
                    saturation=aug_config.get('brightness', 0.1),
                    hue=0.0
                ))

        transform_list.append(transforms.Normalize(
            mean=self.normalize_mean,
            std=self.normalize_std
        ))

        return transforms.Compose(transform_list)

    def get_test_transforms(self) -> transforms.Compose:
        """Get test/validation transforms without augmentation."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std
            )
        ])
