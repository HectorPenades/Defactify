"""Disk cache for pre-computed VLM embeddings."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class EmbeddingCache:
    """
    Save and load pre-computed VLM embeddings per model and split.

    Layout:
        <cache_dir>/<model_name>/
            <split>_embeddings.npy   — (N, D) float32
            <split>_labels_a.npy     — (N,)   int32   Label_A (binary)
            <split>_labels_b.npy     — (N,)   int32   Label_B (multiclass)
            metadata.json            — model info, dim, num_samples, date
    """

    SPLITS = ("train", "validation", "test")

    def __init__(self, cache_dir: str = "./data/cache/embeddings"):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        d = self.root / model_name
        d.mkdir(exist_ok=True)
        return d

    def _paths(self, model_name: str, split: str):
        d = self._model_dir(model_name)
        return {
            "embeddings": d / f"{split}_embeddings.npy",
            "labels_a":   d / f"{split}_labels_a.npy",
            "labels_b":   d / f"{split}_labels_b.npy",
            "metadata":   d / "metadata.json",
        }

    def exists(self, model_name: str, split: str) -> bool:
        """Return True if all three npy files exist for this model + split."""
        p = self._paths(model_name, split)
        return (p["embeddings"].exists()
                and p["labels_a"].exists()
                and p["labels_b"].exists())

    def save(self, model_name: str, split: str,
             embeddings: np.ndarray,
             labels_a: np.ndarray,
             labels_b: np.ndarray) -> None:
        """
        Persist embeddings and both label arrays to disk.

        Args:
            embeddings: (N, D) float32
            labels_a:   (N,)   int32 — binary labels
            labels_b:   (N,)   int32 — multiclass labels
        """
        p = self._paths(model_name, split)
        np.save(p["embeddings"], embeddings.astype(np.float32))
        np.save(p["labels_a"],   labels_a.astype(np.int32))
        np.save(p["labels_b"],   labels_b.astype(np.int32))

        # Update metadata
        meta_path = p["metadata"]
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        meta[split] = {
            "num_samples": int(len(embeddings)),
            "embed_dim":   int(embeddings.shape[1]),
            "computed_at": datetime.now().isoformat(),
        }
        meta["model_name"] = model_name
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, model_name: str, split: str) -> Dict[str, np.ndarray]:
        """
        Load cached embeddings and labels.

        Returns:
            dict with keys: 'embeddings' (N,D), 'labels_a' (N,), 'labels_b' (N,)
        """
        if not self.exists(model_name, split):
            raise FileNotFoundError(
                f"No cache found for model='{model_name}' split='{split}'. "
                f"Run: python scripts/compute_embeddings.py --model {model_name}"
            )
        p = self._paths(model_name, split)
        return {
            "embeddings": np.load(p["embeddings"]),
            "labels_a":   np.load(p["labels_a"]),
            "labels_b":   np.load(p["labels_b"]),
        }

    def load_metadata(self, model_name: str) -> Optional[dict]:
        """Load metadata for a model (or None if not computed yet)."""
        meta_path = self._model_dir(model_name) / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)

    def status(self) -> Dict[str, Dict[str, bool]]:
        """Return cache status for all known models and splits."""
        from src.processing.embeddings import EXTRACTORS
        result = {}
        for model_name in EXTRACTORS:
            result[model_name] = {
                split: self.exists(model_name, split)
                for split in self.SPLITS
            }
        return result
