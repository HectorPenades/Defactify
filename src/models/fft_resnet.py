"""ResNet50 adapted for single-channel or 3-channel FFT input."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from src.models.base import BaseModel


class FFTResNet50(BaseModel):
    """ResNet50 for FFT magnitude images.

    Supports:
    - 1-channel input (fft_grayscale): Conv1 kernel re-initialized to accept 1 channel
    - 3-channel input (fft_perchannel): same as RGB but weights averaged across channels
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict. Reads:
                model.pretrained (bool)
                model.dropout (float)
                model.num_classes (int)
                dataset.mode ('fft_grayscale' | 'fft_perchannel')
        """
        super().__init__(config)
        self.pretrained = config.get('model', {}).get('pretrained', True)
        self.dropout_rate = config.get('model', {}).get('dropout', 0.5)
        mode = config.get('dataset', {}).get('mode', 'fft_grayscale')
        self.in_channels = 1 if mode == 'fft_grayscale' else 3

        backbone = models.resnet50(pretrained=self.pretrained)

        if self.in_channels == 1:
            # Adapt Conv1: average pretrained RGB weights across channels → 1 channel
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            if self.pretrained:
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv

        # Replace classifier head
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(num_ftrs, self.num_classes)
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) where C=1 for grayscale or C=3 for per-channel
        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2048-d features before the classifier head."""
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)
