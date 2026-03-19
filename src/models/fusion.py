"""Late Fusion model: RGB ResNet50 + FFT ResNet50 → MLP classifier."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from src.models.base import BaseModel


class _FeatureExtractor(nn.Module):
    """ResNet50 backbone that returns 2048-d pooled features (no classifier)."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)

        if in_channels == 1:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            if pretrained:
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv

        # Remove the FC head — keep everything up to avgpool
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.feature_dim = backbone.fc.in_features  # 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 2048)


class LateFusionModel(BaseModel):
    """
    Two-branch late fusion model.

    Branch 1 (RGB):  ResNet50 → 2048-d features
    Branch 2 (FFT):  ResNet50 (1-ch or 3-ch) → 2048-d features
    Head:            MLP(4096 → 512 → num_classes)

    forward(x_rgb, x_fft) → logits (B, num_classes)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict. Reads:
                model.pretrained (bool)
                model.dropout (float)
                model.num_classes (int)
                dataset.mode ('fusion_late')
                dataset.fft_channels (int, default 1)
        """
        super().__init__(config)
        pretrained = config.get('model', {}).get('pretrained', True)
        dropout_rate = config.get('model', {}).get('dropout', 0.5)
        fft_channels = config.get('dataset', {}).get('fft_channels', 1)

        self.rgb_branch = _FeatureExtractor(in_channels=3, pretrained=pretrained)
        self.fft_branch = _FeatureExtractor(in_channels=fft_channels, pretrained=pretrained)

        feat_dim = self.rgb_branch.feature_dim + self.fft_branch.feature_dim  # 4096

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x_rgb: torch.Tensor, x_fft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB images (B, 3, H, W)
            x_fft: FFT images (B, 1, H, W) or (B, 3, H, W)
        Returns:
            Logits (B, num_classes)
        """
        feat_rgb = self.rgb_branch(x_rgb)
        feat_fft = self.fft_branch(x_fft)
        combined = torch.cat([feat_rgb, feat_fft], dim=1)
        return self.classifier(combined)
