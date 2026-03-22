#!/usr/bin/env python
"""Main experiment orchestrator."""

import sys
import torch
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import yaml

# Add parent directory to path so we can import src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dataset import DefactifyDataset
from src.data.transforms import ImagePreprocessor
from src.models.rgb_resnet import RGBResNet50
from src.models.fft_resnet import FFTResNet50
from src.models.fusion import LateFusionModel
from src.models.vit_classifier import ViTClassifier
from src.training.trainer import DefaultTrainer
from src.inference.evaluation import Evaluator
from src.utils.logging import ExperimentLogger
from src.utils.reproducibility import ReproducibilityManager
from src.utils.results_tracker import record_result


class ExperimentRunner:
    """Runner for ML experiments."""

    def __init__(self, config_path: str, resume: bool = False):
        """
        Args:
            config_path: Path to experiment config YAML
            resume: If True, continue from checkpoint_last.pth
        """
        self.config = load_config(config_path)
        self.resume = resume
        self.setup_directories()

    def setup_directories(self) -> None:
        """Setup result directories."""
        exp_name = self.config['experiment_name']
        results_dir = Path(self.config['output']['results_dir'])
        self.exp_dir = results_dir / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run complete experiment."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {self.config['experiment_name']}")
        print(f"{'='*60}\n")

        # Setup reproducibility
        ReproducibilityManager.set_seed(self.config['seed'])

        # Initialize logger (with W&B integration)
        logger = ExperimentLogger(self.exp_dir, config=self.config, use_wandb=True)
        logger.save_config(self.config)
        logger.log_message(f"Experiment: {self.config['experiment_name']}")

        try:
            # 1. Load datasets
            logger.log_message("Loading datasets...")
            train_dataset = DefactifyDataset(
                self.config,
                split='train',
                transform=ImagePreprocessor(self.config).get_train_transforms()
            )
            val_dataset = DefactifyDataset(
                self.config,
                split='validation',
                transform=ImagePreprocessor(self.config).get_test_transforms()
            )
            test_dataset = DefactifyDataset(
                self.config,
                split='test',
                transform=ImagePreprocessor(self.config).get_test_transforms()
            )

            logger.log_message(f"Train size: {len(train_dataset)}")
            logger.log_message(f"Val size: {len(val_dataset)}")
            logger.log_message(f"Test size: {len(test_dataset)}")

            # 2. Create data loaders
            logger.log_message("Creating data loaders...")
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'] * 2,
                shuffle=False,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.config['training']['batch_size'] * 2,
                shuffle=False,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True
            )

            # 3. Build model
            logger.log_message("Building model...")
            model = self._build_model()
            num_params = sum(p.numel() for p in model.parameters())
            logger.log_message(f"Model parameters: {num_params:,}")

            # 4. Train
            logger.log_message("Starting training...")
            trainer = DefaultTrainer(
                model, train_loader, val_loader, self.config, logger
            )

            # Load resume state if requested
            resume_state = None
            last_ckpt = self.exp_dir / 'checkpoint_last.pth'
            if self.resume:
                if last_ckpt.exists():
                    resume_state = torch.load(last_ckpt, map_location=self.config['device'])
                    model.load_state_dict(resume_state['model_state_dict'])
                else:
                    logger.log_message(
                        "WARNING: --resume requested but checkpoint_last.pth not found. "
                        "Starting from scratch."
                    )

            trainer.train(
                num_epochs=self.config['training']['num_epochs'],
                patience=self.config['training']['early_stopping_patience'],
                resume_state=resume_state
            )

            # 5. Evaluate on test set
            logger.log_message("Evaluating on test set...")
            evaluator = Evaluator(model, device=self.config['device'])

            # Load best checkpoint
            best_epoch = None
            total_epochs = None
            checkpoint_path = self.exp_dir / 'checkpoint_best.pth'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                best_epoch = checkpoint['epoch']
                logger.log_message(f"Loaded best checkpoint from epoch {best_epoch + 1}")
            last_ckpt_path = self.exp_dir / 'checkpoint_last.pth'
            if last_ckpt_path.exists():
                last_ckpt = torch.load(last_ckpt_path)
                total_epochs = last_ckpt['epoch'] + 1

            test_results = evaluator.evaluate_on_split(
                test_loader,
                self.config['model']['num_classes'],
                task=self.config['task']
            )

            # Save predictions
            pred_path = self.exp_dir / 'predictions.csv'
            evaluator.save_predictions(test_results, str(pred_path))

            # Save final metrics
            logger.save_final_metrics(
                test_results['metrics'],
                best_epoch=best_epoch,
                total_epochs=total_epochs,
            )

            # Update global comparison table
            duration_s = (datetime.now() - logger.start_time).total_seconds()
            record_result(
                config=self.config,
                test_metrics=test_results['metrics'],
                checkpoint_path=str(self.exp_dir / 'checkpoint_best.pth'),
                duration_s=duration_s,
            )

            # Log summary
            logger.log_message("\n" + "="*60)
            logger.log_message("FINAL RESULTS")
            logger.log_message("="*60)
            logger.log_message(f"Accuracy: {test_results['metrics']['accuracy']:.4f}")
            logger.log_message(f"F1 (macro): {test_results['metrics']['f1_macro']:.4f}")
            logger.log_message(f"F1 (weighted): {test_results['metrics']['f1_weighted']:.4f}")

            print(f"\n✅ Experiment completed successfully!")
            print(f"📁 Results saved to: {self.exp_dir}")

        except Exception as e:
            logger.log_message(f"ERROR: {str(e)}")
            print(f"\n❌ Experiment failed with error: {e}")
            raise

    def _build_model(self) -> torch.nn.Module:
        """Build model based on config."""
        architecture = self.config['model']['architecture']

        if architecture == 'rgb_resnet50':
            return RGBResNet50(self.config)
        elif architecture == 'fft_resnet50':
            return FFTResNet50(self.config)
        elif architecture == 'late_fusion':
            return LateFusionModel(self.config)
        elif architecture == 'vit':
            return ViTClassifier(self.config)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Defactify experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config YAML')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate config without running')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint_last.pth')
    args = parser.parse_args()

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    if args.dry_run:
        config = load_config(str(config_path))
        print(f"✅ Config valid: {config['experiment_name']}")
        return

    # Run experiment
    runner = ExperimentRunner(str(config_path), resume=args.resume)
    runner.run()


if __name__ == '__main__':
    main()
