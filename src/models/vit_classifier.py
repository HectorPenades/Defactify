"""Vision Transformer (ViT-B/16) classifier for binary and multiclass tasks."""

import torch
import torch.nn as nn
from typing import Dict, Any
from transformers import ViTModel, ViTConfig

from src.models.base import BaseModel


class ViTClassifier(BaseModel):
    """ViT-Base/16 from HuggingFace transformers.

    Uses the CLS token representation from the last hidden state as input to
    a linear classifier head, identical pattern to RGBResNet50.

    Supported in config:
        model.pretrained  : bool  — load ImageNet-21k weights (default True)
        model.dropout     : float — dropout before classifier (default 0.1)
        model.num_classes : int   — number of output classes (default 2)

    Input: (B, 3, 224, 224) tensor normalised with ViT preprocessing
           mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    Output: (B, num_classes) logits
    """

    MODEL_ID = 'google/vit-base-patch16-224-in21k'

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pretrained = config.get('model', {}).get('pretrained', True)
        self.dropout_rate = config.get('model', {}).get('dropout', 0.1)

        if self.pretrained:
            self.vit = ViTModel.from_pretrained(self.MODEL_ID)
        else:
            # Same architecture, random weights
            vit_config = ViTConfig.from_pretrained(self.MODEL_ID)
            self.vit = ViTModel(vit_config)

        hidden_size = self.vit.config.hidden_size  # 768 for ViT-B/16
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(hidden_size, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)

        Returns:
            logits: (B, num_classes)
        """
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]   # (B, 768)
        return self.classifier(cls_token)
