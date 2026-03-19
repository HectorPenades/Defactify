"""RGB ResNet50 model for binary and multiclass classification."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from src.models.base import BaseModel


class RGBResNet50(BaseModel):
    """ResNet50 model for RGB images."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RGB ResNet50.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.pretrained = config.get('model', {}).get('pretrained', True)
        self.dropout_rate = config.get('model', {}).get('dropout', 0.5)

        # Load ResNet50
        self.backbone = models.resnet50(pretrained=self.pretrained)

        # Get number of input features
        num_ftrs = self.backbone.fc.in_features

        # Replace final FC layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(num_ftrs, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)
