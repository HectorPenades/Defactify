#!/usr/bin/env python
"""
Standalone test evaluation script.

Loads a saved checkpoint and evaluates it on the test split.
Results are printed, saved to <results_dir>/<exp_name>/test_eval.json,
and appended to results/comparison_table.csv.

Usage:
    # Use the checkpoint saved by run_experiments.py (default path):
    python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml

    # Use a custom checkpoint path:
    python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml \
        --checkpoint path/to/checkpoint_best.pth

    # Print metrics only, do not update the comparison table:
    python scripts/run_test.py --config configs/experiments/01_rgb_baseline_binary.yaml \
        --no-record
"""

import sys
import json
import argparse
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
from src.utils.results_tracker import record_result


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


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved checkpoint on the test split')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config YAML')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth file. '
                             'Defaults to results/experiments/<exp_name>/checkpoint_best.pth')
    parser.add_argument('--no-record', action='store_true',
                        help='Do not update the comparison table')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    exp_name = config['experiment_name']
    results_dir = Path(config['output']['results_dir']) / exp_name

    # Resolve checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = results_dir / 'checkpoint_best.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Run the experiment first with scripts/run_experiments.py")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Test evaluation: {exp_name}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"{'='*60}\n")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = DefactifyDataset(
        config,
        split='test',
        transform=ImagePreprocessor(config).get_test_transforms()
    )
    print(f"Test size: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    # Build and load model
    print("Building model...")
    model = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=config.get('device', 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint.get('epoch', '?')
    print(f"Loaded checkpoint from epoch {best_epoch}")

    # Evaluate
    start = datetime.now()
    evaluator = Evaluator(model, device=config.get('device', 'cuda:0'))
    results = evaluator.evaluate_on_split(
        test_loader,
        config['model']['num_classes'],
        task=config.get('task', 'multiclass')
    )
    duration_s = (datetime.now() - start).total_seconds()

    metrics = results['metrics']

    # Print results
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    # Save predictions + metrics to experiment directory
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / 'test_eval_predictions.csv'
    evaluator.save_predictions(results, str(pred_path))

    eval_path = results_dir / 'test_eval.json'
    with open(eval_path, 'w') as f:
        json.dump({
            'test_metrics': metrics,
            'checkpoint': str(checkpoint_path),
            'epoch': best_epoch,
            'evaluated_at': datetime.now().isoformat(),
            'duration_s': duration_s,
        }, f, indent=2)
    print(f"\nMetrics saved to: {eval_path}")

    # Update comparison table
    if not args.no_record:
        record_result(
            config=config,
            test_metrics=metrics,
            checkpoint_path=str(checkpoint_path),
            duration_s=duration_s,
        )


if __name__ == '__main__':
    main()
