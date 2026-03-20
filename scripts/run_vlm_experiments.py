"""Run VLM classifier sweep experiments.

Loads pre-computed embeddings from cache and runs a sweep of classifiers.
Results are saved under results/experiments/<experiment_name>/.

Usage:
    python scripts/run_vlm_experiments.py --config configs/experiments/07_clip_multiclass.yaml
    python scripts/run_vlm_experiments.py --config configs/experiments/10_vlm_ensemble_multiclass.yaml
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import yaml

from src.utils.embedding_cache import EmbeddingCache
from src.classifiers.sweep import run_sweep
from src.utils.results_tracker import record_result


def _load_config(path: str) -> dict:
    """Load experiment config, merging with base_config."""
    base_path = Path("configs/base_config.yaml")
    with open(base_path) as f:
        config = yaml.safe_load(f)
    with open(path) as f:
        override = yaml.safe_load(f)
    _deep_merge(config, override)
    return config


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _load_embeddings(cache: EmbeddingCache, model_name: str):
    """Load train/val/test embeddings and labels for one VLM model."""
    splits = {}
    for split in ("train", "validation", "test"):
        data = cache.load(model_name, split)
        splits[split] = data
    return splits


def _concat_embeddings(cache: EmbeddingCache, model_names: list):
    """Concatenate embeddings from multiple VLM models (ensemble)."""
    all_splits = {s: {"embeddings": [], "labels_a": None, "labels_b": None}
                  for s in ("train", "validation", "test")}
    for model_name in model_names:
        for split in ("train", "validation", "test"):
            data = cache.load(model_name, split)
            all_splits[split]["embeddings"].append(data["embeddings"])
            all_splits[split]["labels_a"] = data["labels_a"]   # same for all models
            all_splits[split]["labels_b"] = data["labels_b"]
    # Concatenate along feature dimension
    for split in all_splits:
        all_splits[split]["embeddings"] = np.concatenate(
            all_splits[split]["embeddings"], axis=1)
    return all_splits


def main():
    parser = argparse.ArgumentParser(description="Run VLM classifier experiments")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--cache-dir", default="./data/cache/embeddings",
        help="Embedding cache directory",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device from config",
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="Skip updating the comparison table",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    device = args.device or config.get("device", "cuda:0")

    exp_name = config.get("experiment_name", "vlm_exp")
    task     = config.get("task", "multiclass")
    num_classes = config.get("model", {}).get("num_classes", 6)
    vlm_cfg  = config.get("vlm", {})
    model_names = vlm_cfg.get("models", [config.get("model", {}).get("vlm_backbone", "clip-vit-b32")])

    results_dir = Path(config.get("output", {}).get("results_dir", "./results/experiments"))
    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"Task       : {task}  ({num_classes} classes)")
    print(f"VLM models : {model_names}")
    print(f"Device     : {device}")
    print(f"Output     : {exp_dir}")
    print(f"{'='*60}\n")

    # ── Load embeddings ──────────────────────────────────────────────────────
    cache = EmbeddingCache(args.cache_dir)

    if len(model_names) == 1:
        splits = _load_embeddings(cache, model_names[0])
    else:
        print(f"  Concatenating embeddings from {len(model_names)} models...")
        splits = _concat_embeddings(cache, model_names)

    label_key = "labels_a" if task == "binary" else "labels_b"

    X_train = splits["train"]["embeddings"]
    y_train = splits["train"][label_key]
    X_val   = splits["validation"]["embeddings"]
    y_val   = splits["validation"][label_key]
    X_test  = splits["test"]["embeddings"]
    y_test  = splits["test"][label_key]

    print(f"  Embedding dim : {X_train.shape[1]}")
    print(f"  Train samples : {len(X_train)}")
    print(f"  Val   samples : {len(X_val)}")
    print(f"  Test  samples : {len(X_test)}")
    print()

    # ── Run classifier sweep ─────────────────────────────────────────────────
    t0 = time.time()
    sweep_result = run_sweep(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        num_classes=num_classes,
        task=task,
        config=config,
        exp_dir=exp_dir,
        device=device,
    )
    duration_s = time.time() - t0

    # ── Print summary ────────────────────────────────────────────────────────
    best_name    = sweep_result["best_classifier"]
    best_val_f1  = sweep_result["best_val_f1_macro"]
    best_test_m  = sweep_result["best_test_metrics"]

    print(f"\n{'='*60}")
    print(f"Best classifier : {best_name}")
    print(f"Val  F1-macro   : {best_val_f1:.4f}")
    if best_test_m:
        print(f"Test F1-macro   : {best_test_m.get('f1_macro', 'N/A'):.4f}")
        print(f"Test Accuracy   : {best_test_m.get('accuracy', 'N/A'):.4f}")
        print(f"Test Bal.Acc    : {best_test_m.get('balanced_accuracy', 'N/A'):.4f}")
    print(f"Duration        : {duration_s:.1f}s")
    print(f"{'='*60}\n")

    # ── Update comparison table ──────────────────────────────────────────────
    if not args.no_record and best_test_m:
        # Inject arch info for the tracker
        config.setdefault("model", {})["architecture"] = f"vlm_{best_name}"
        config.setdefault("dataset", {})["mode"] = "+".join(model_names)
        record_result(
            config=config,
            test_metrics=best_test_m,
            checkpoint_path=str(exp_dir / "models" / f"{best_name}_best.joblib"),
            duration_s=duration_s,
        )

    print(f"\nResults saved to: {exp_dir}")


if __name__ == "__main__":
    main()
