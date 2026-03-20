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

        pre_tensor = []   # applied before ToTensor (PIL operations)
        post_tensor = []  # applied after ToTensor

        if aug_config.get('enabled', True):
            # Random resized crop (replaces fixed resize — only if enabled)
            if aug_config.get('random_resized_crop', False):
                scale_min = aug_config.get('random_resized_crop_scale_min', 0.7)
                pre_tensor.append(transforms.RandomResizedCrop(
                    self.image_size, scale=(scale_min, 1.0)
                ))

            # Horizontal flip
            if aug_config.get('random_flip', 0) > 0:
                pre_tensor.append(transforms.RandomHorizontalFlip(
                    p=aug_config.get('random_flip', 0.5)
                ))

            # Random rotation
            if aug_config.get('rotation', 0) > 0:
                pre_tensor.append(transforms.RandomRotation(
                    degrees=aug_config.get('rotation', 0)
                ))

            # Color jitter
            if aug_config.get('color_jitter', 0) > 0:
                pre_tensor.append(transforms.ColorJitter(
                    brightness=aug_config.get('brightness', 0.1),
                    contrast=aug_config.get('contrast', 0.1),
                    saturation=aug_config.get('saturation', aug_config.get('brightness', 0.1)),
                    hue=aug_config.get('hue', 0.0),
                ))

            # Random grayscale
            if aug_config.get('grayscale', 0) > 0:
                pre_tensor.append(transforms.RandomGrayscale(
                    p=aug_config.get('grayscale', 0)
                ))

            # Gaussian blur (PIL, before ToTensor)
            if aug_config.get('blur', 0) > 0:
                pre_tensor.append(transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                ], p=aug_config.get('blur', 0)))

        transform_list = pre_tensor + [transforms.ToTensor()]

        transform_list.append(transforms.Normalize(
            mean=self.normalize_mean,
            std=self.normalize_std
        ))

        # Random erasing (after ToTensor + Normalize)
        if aug_config.get('enabled', True) and aug_config.get('random_erasing', 0) > 0:
            transform_list.append(transforms.RandomErasing(
                p=aug_config.get('random_erasing', 0),
                scale=(0.02, 0.2),
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
