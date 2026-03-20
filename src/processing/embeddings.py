"""VLM feature extractors — CLIP, DINOv2, SigLIP (frozen, no gradients)."""

import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# Base extractor
# ──────────────────────────────────────────────────────────────────────────────

class VLMExtractor(ABC):
    """Abstract base for frozen VLM image feature extractors."""

    MODEL_ID: str = ""
    EMBED_DIM: int = 0

    def __init__(self, device: str = "cuda:0", half_precision: bool = True):
        self.device = device
        self.dtype = torch.float16 if (half_precision and "cuda" in device) else torch.float32
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load model and processor."""

    @abstractmethod
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract embeddings for a batch of PIL images.

        Returns:
            np.ndarray of shape (B, embed_dim), float32
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.EMBED_DIM}, device={self.device})"


# ──────────────────────────────────────────────────────────────────────────────
# CLIP ViT-B/32
# ──────────────────────────────────────────────────────────────────────────────

class CLIPExtractor(VLMExtractor):
    """OpenAI CLIP ViT-B/32 — 512-dim image embeddings."""

    MODEL_ID = "openai/clip-vit-base-patch32"
    EMBED_DIM = 512

    def _load_model(self) -> None:
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID, use_safetensors=True).to(self.device)
        if self.dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalize
        return feats.float().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# DINOv2-L
# ──────────────────────────────────────────────────────────────────────────────

class DINOv2Extractor(VLMExtractor):
    """Meta DINOv2-Large — 1024-dim CLS token embeddings."""

    MODEL_ID = "facebook/dinov2-large"
    EMBED_DIM = 1024

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, Dinov2Model
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = Dinov2Model.from_pretrained(self.MODEL_ID, use_safetensors=True).to(self.device)
        if self.dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]   # CLS token
        return feats.float().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# SigLIP SO400M
# ──────────────────────────────────────────────────────────────────────────────

class SigLIPExtractor(VLMExtractor):
    """Google SigLIP SO400M patch14@384 — 1152-dim pooled embeddings."""

    MODEL_ID = "google/siglip-so400m-patch14-384"
    EMBED_DIM = 1152

    def _load_model(self) -> None:
        from transformers import AutoProcessor, SiglipVisionModel
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = SiglipVisionModel.from_pretrained(self.MODEL_ID, use_safetensors=True).to(self.device)
        if self.dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            feats = outputs.pooler_output
        return feats.float().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

EXTRACTORS = {
    "clip-vit-b32":       CLIPExtractor,
    "dinov2-l":           DINOv2Extractor,
    "siglip-so400m":      SigLIPExtractor,
}


def get_extractor(model_name: str, device: str = "cuda:0",
                  half_precision: bool = True) -> VLMExtractor:
    """
    Instantiate a VLM extractor by name.

    Args:
        model_name: 'clip-vit-b32' | 'dinov2-l' | 'siglip-so400m'
        device: torch device string
        half_precision: Use float16 on CUDA (saves VRAM, negligible accuracy loss)
    """
    if model_name not in EXTRACTORS:
        raise ValueError(f"Unknown VLM model: {model_name}. "
                         f"Choose from: {list(EXTRACTORS)}")
    return EXTRACTORS[model_name](device=device, half_precision=half_precision)
