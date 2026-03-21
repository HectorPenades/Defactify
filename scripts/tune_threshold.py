"""Threshold tuning for binary classification.

Finds the optimal decision threshold using the validation set
and re-evaluates the test set with that threshold.
No retraining needed — works with any existing binary checkpoint.

Usage:
    python scripts/tune_threshold.py --config configs/experiments/01_rgb_baseline_binary.yaml
    python scripts/tune_threshold.py --config configs/experiments/13_late_fusion_perchannel_binary.yaml
    python scripts/tune_threshold.py --config configs/experiments/01_rgb_baseline_binary.yaml \\
        --metric f1_macro      # optimise threshold for f1_macro (default)
    python scripts/tune_threshold.py --config configs/experiments/01_rgb_baseline_binary.yaml \\
        --metric balanced_accuracy
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dataset import DefactifyDataset
from src.data.transforms import ImagePreprocessor
from src.models.rgb_resnet import RGBResNet50
from src.models.fft_resnet import FFTResNet50
from src.models.fusion import LateFusionModel
from src.inference.evaluation import Evaluator
from src.training.metrics import MetricsComputer


def build_model(config: dict) -> torch.nn.Module:
    arch = config['model']['architecture']
    if arch == 'rgb_resnet50':
        return RGBResNet50(config)
    elif arch == 'fft_resnet50':
        return FFTResNet50(config)
    elif arch == 'late_fusion':
        return LateFusionModel(config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply custom threshold to binary probabilities.
    probs: (N, 2) — col 0 = P(real), col 1 = P(AI)
    Returns predictions with: real if P(real) >= threshold, else AI.
    """
    return (probs[:, 0] < threshold).astype(int)


def evaluate_with_threshold(probs: np.ndarray, labels: np.ndarray,
                             threshold: float) -> dict:
    """Compute metrics applying a custom threshold."""
    preds = apply_threshold(probs, threshold)
    # Build fake probs matrix compatible with MetricsComputer
    # (it reads argmax anyway when threshold is embedded in preds)
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)

    acc      = accuracy_score(labels, preds)
    bal_acc  = balanced_accuracy_score(labels, preds)
    f1_mac   = f1_score(labels, preds, average='macro', zero_division=0)
    f1_wei   = f1_score(labels, preds, average='weighted', zero_division=0)
    prec     = precision_score(labels, preds, average='macro', zero_division=0)
    rec      = recall_score(labels, preds, average='macro', zero_division=0)
    roc_auc  = roc_auc_score(labels, probs[:, 1])
    cm       = confusion_matrix(labels, preds).tolist()

    # Per-class
    per_class = {}
    for idx, name in {0: 'real', 1: 'ai_generated'}.items():
        per_class[name] = {
            'precision': float(precision_score(labels, preds, labels=[idx],
                                               average='macro', zero_division=0)),
            'recall':    float(recall_score(labels, preds, labels=[idx],
                                            average='macro', zero_division=0)),
            'f1':        float(f1_score(labels, preds, labels=[idx],
                                        average='macro', zero_division=0)),
            'support':   int((labels == idx).sum()),
        }

    return {
        'accuracy':          float(acc),
        'balanced_accuracy': float(bal_acc),
        'f1_macro':          float(f1_mac),
        'f1_weighted':       float(f1_wei),
        'precision':         float(prec),
        'recall':            float(rec),
        'roc_auc':           float(roc_auc),
        'confusion_matrix':  cm,
        'per_class':         per_class,
    }


def collect_probs(evaluator: Evaluator, loader, device: str):
    """Run model on a DataLoader and collect probabilities + labels."""
    all_probs, all_labels = [], []
    evaluator.model.eval()
    with torch.no_grad():
        for batch in loader:
            _, probs = evaluator.predict_batch(batch)
            all_probs.extend(probs)
            all_labels.extend(batch['label'].numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Tune decision threshold for binary task')
    parser.add_argument('--config',     required=True,  help='Experiment YAML config')
    parser.add_argument('--checkpoint', default=None,   help='Checkpoint .pth (default: checkpoint_best.pth)')
    parser.add_argument('--metric',     default='f1_macro',
                        choices=['f1_macro', 'balanced_accuracy', 'f1_real_recall'],
                        help='Metric to optimise on validation set (default: f1_macro)')
    parser.add_argument('--steps',      type=int, default=99,
                        help='Number of threshold candidates between 0.01 and 0.99 (default: 99)')
    args = parser.parse_args()

    config = load_config(args.config)

    if config.get('task') != 'binary':
        print("ERROR: threshold tuning only applies to binary experiments.")
        sys.exit(1)

    exp_name    = config['experiment_name']
    device      = config.get('device', 'cuda:0')
    results_dir = Path(config['output']['results_dir']) / exp_name

    ckpt_path = Path(args.checkpoint) if args.checkpoint else results_dir / 'checkpoint_best.pth'
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Threshold tuning: {exp_name}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Optimising : {args.metric} on validation set")
    print(f"{'='*60}\n")

    # ── Load model ───────────────────────────────────────────────────────────
    model = build_model(config)
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    evaluator = Evaluator(model, device=device)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')})")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    transform   = ImagePreprocessor(config).get_test_transforms()
    num_workers = config.get('num_workers', 4)
    batch_size  = config['training']['batch_size'] * 2

    print("Collecting validation probabilities...")
    val_ds     = DefactifyDataset(config, split='validation', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    val_probs, val_labels = collect_probs(evaluator, val_loader, device)

    print("Collecting test probabilities...")
    test_ds     = DefactifyDataset(config, split='test', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    test_probs, test_labels = collect_probs(evaluator, test_loader, device)

    # ── Sweep thresholds on validation ───────────────────────────────────────
    thresholds  = np.linspace(0.01, 0.99, args.steps)
    best_thresh = 0.5
    best_score  = -1.0

    print(f"\nSweeping {len(thresholds)} thresholds on validation set...")
    for t in thresholds:
        m = evaluate_with_threshold(val_probs, val_labels, t)
        if args.metric == 'f1_real_recall':
            score = m['per_class']['real']['recall']
        else:
            score = m[args.metric]
        if score > best_score:
            best_score  = score
            best_thresh = t

    print(f"\n  Default threshold (0.50): val {args.metric} = "
          f"{evaluate_with_threshold(val_probs, val_labels, 0.5)[args.metric]:.4f}")
    print(f"  Best   threshold ({best_thresh:.2f}): val {args.metric} = {best_score:.4f}")

    # ── Evaluate test with default vs best threshold ──────────────────────────
    default_m = evaluate_with_threshold(test_probs, test_labels, 0.5)
    best_m    = evaluate_with_threshold(test_probs, test_labels, best_thresh)

    print(f"\n{'='*60}")
    print("TEST SET — Default threshold (0.50) vs Best threshold ({:.2f})".format(best_thresh))
    print(f"{'='*60}")
    metrics_to_show = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc']
    print(f"  {'Metric':<22} {'Default':>10} {'Best':>10} {'Δ':>8}")
    print(f"  {'-'*52}")
    for k in metrics_to_show:
        d, b = default_m[k], best_m[k]
        print(f"  {k:<22} {d:>10.4f} {b:>10.4f} {b-d:>+8.4f}")

    print(f"\n  Per-class (best threshold):")
    for cls, vals in best_m['per_class'].items():
        print(f"    {cls:<15}  precision={vals['precision']:.4f}  "
              f"recall={vals['recall']:.4f}  f1={vals['f1']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        'experiment':       exp_name,
        'checkpoint':       str(ckpt_path),
        'optimised_metric': args.metric,
        'best_threshold':   round(float(best_thresh), 4),
        'val': {
            'default_threshold_0.5': evaluate_with_threshold(val_probs, val_labels, 0.5),
            f'best_threshold_{best_thresh:.2f}': evaluate_with_threshold(val_probs, val_labels, best_thresh),
        },
        'test': {
            'default_threshold_0.5': default_m,
            f'best_threshold_{best_thresh:.2f}': best_m,
        },
        'evaluated_at': datetime.now().isoformat(),
    }

    out_path = results_dir / 'threshold_tuning.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"\nTo use threshold {best_thresh:.2f} in production: "
          f"predict 'real' if P(real) >= {best_thresh:.2f}, else 'AI'")


if __name__ == '__main__':
    main()
