"""Pre-compute and cache VLM embeddings for all dataset splits.

Usage:
    python scripts/compute_embeddings.py --model clip-vit-b32
    python scripts/compute_embeddings.py --model all --splits all
    python scripts/compute_embeddings.py --model dinov2-l --force
"""

import argparse
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from src.processing.embeddings import EXTRACTORS, get_extractor
from src.utils.embedding_cache import EmbeddingCache


HF_REPO = "Rajarshi-Roy-research/Defactify_Image_Dataset"
SPLIT_MAP = {
    "train":      "train",
    "validation": "validation",
    "test":       "test",
}


def _to_pil(image_field) -> Image.Image:
    """Convert HF image field to PIL RGB."""
    if isinstance(image_field, dict):
        # HF datasets Image feature returns dict with 'bytes' or 'path'
        img = Image.open(image_field["path"]) if "path" in image_field else \
              Image.open(image_field["bytes"])
    elif isinstance(image_field, Image.Image):
        img = image_field
    else:
        img = Image.fromarray(image_field)
    return img.convert("RGB")


def compute_split(
    model_name: str,
    split: str,
    batch_size: int,
    device: str,
    cache: EmbeddingCache,
    force: bool,
) -> None:
    if cache.exists(model_name, split) and not force:
        print(f"  [{model_name}] {split}: already cached, skipping.")
        return

    print(f"  [{model_name}] {split}: loading extractor...")
    extractor = get_extractor(model_name, device=device, half_precision=True)

    print(f"  [{model_name}] {split}: loading dataset split...")
    ds = load_dataset(HF_REPO, split=split)
    n = len(ds)
    print(f"  [{model_name}] {split}: {n} samples, batch_size={batch_size}")

    all_embeddings = []
    all_labels_a   = []
    all_labels_b   = []

    t0 = time.time()
    for start in tqdm(range(0, n, batch_size), desc=f"{model_name}/{split}"):
        batch = ds[start: start + batch_size]
        images = [_to_pil(img) for img in batch["Image"]]
        embs   = extractor.extract_batch(images)   # (B, D) float32
        all_embeddings.append(embs)
        all_labels_a.extend(batch["Label_A"])
        all_labels_b.extend(batch["Label_B"])

    embeddings = np.concatenate(all_embeddings, axis=0)   # (N, D) float32
    labels_a   = np.array(all_labels_a, dtype=np.int32)
    labels_b   = np.array(all_labels_b, dtype=np.int32)

    cache.save(model_name, split, embeddings, labels_a, labels_b)
    elapsed = time.time() - t0
    print(f"  [{model_name}] {split}: saved {embeddings.shape} in {elapsed:.1f}s")

    # Free GPU memory before next model
    del extractor
    try:
        import torch
        if "cuda" in device:
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VLM embeddings")
    parser.add_argument(
        "--model", default="all",
        choices=list(EXTRACTORS) + ["all"],
        help="VLM model name (default: all)",
    )
    parser.add_argument(
        "--splits", default="all",
        choices=["train", "validation", "test", "all"],
        help="Dataset split(s) to compute (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    parser.add_argument(
        "--cache-dir", default="./data/cache/embeddings",
        help="Embedding cache directory",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cache already exists",
    )
    args = parser.parse_args()

    models = list(EXTRACTORS) if args.model == "all" else [args.model]
    splits = list(SPLIT_MAP)   if args.splits == "all" else [args.splits]

    cache = EmbeddingCache(args.cache_dir)

    print(f"Models : {models}")
    print(f"Splits : {splits}")
    print(f"Device : {args.device}")
    print(f"Cache  : {args.cache_dir}")
    print()

    for model_name in models:
        for split in splits:
            compute_split(
                model_name=model_name,
                split=split,
                batch_size=args.batch_size,
                device=args.device,
                cache=cache,
                force=args.force,
            )

    # Print final status
    print("\n── Cache status ───────────────────────────────────────────────")
    status = cache.status()
    for m, splits_status in status.items():
        row = "  ".join(f"{s}: {'OK' if ok else '--'}" for s, ok in splits_status.items())
        print(f"  {m:<25} {row}")


if __name__ == "__main__":
    main()
