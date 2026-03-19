"""Experiment logging and metrics tracking with W&B integration."""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ExperimentLogger:
    """Logger for experiment metrics and results."""

    def __init__(self, log_dir: Path, config: Optional[Dict[str, Any]] = None,
                 use_wandb: bool = True):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            config: Configuration dictionary for W&B
            use_wandb: Enable Weights & Biases logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {'train': [], 'val': []}
        self.start_time = datetime.now()
        self.use_wandb = use_wandb and HAS_WANDB

        # Initialize W&B if available and requested
        if self.use_wandb and config:
            exp_name = config.get('experiment_name', 'experiment')
            try:
                wandb.init(
                    project="real-vs-synthetic",
                    name=exp_name,
                    config=config,
                    reinit=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize W&B: {e}")
                self.use_wandb = False

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float]) -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
        """
        self.metrics['train'].append({
            'epoch': epoch,
            **train_metrics
        })
        self.metrics['val'].append({
            'epoch': epoch,
            **val_metrics
        })

        # Log to W&B if available
        if self.use_wandb:
            wandb_log = {f"train/{k}": v for k, v in train_metrics.items()}
            wandb_log.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb_log['epoch'] = epoch
            wandb.log(wandb_log)

    def save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics_path = self.log_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def save_final_metrics(self, test_metrics: Dict[str, Any],
                           best_epoch: Optional[int] = None,
                           total_epochs: Optional[int] = None) -> None:
        """
        Save final test metrics.

        Args:
            test_metrics: Dictionary of test metrics
            best_epoch: Epoch number (0-indexed) of the best checkpoint used for test
            total_epochs: Total epochs trained before stopping
        """
        final_metrics = {
            'test': test_metrics,
            'experiment_duration_seconds': (
                datetime.now() - self.start_time
            ).total_seconds()
        }
        if best_epoch is not None:
            final_metrics['best_epoch'] = best_epoch + 1        # 1-indexed for readability
            final_metrics['best_epoch_0indexed'] = best_epoch
        if total_epochs is not None:
            final_metrics['total_epochs_trained'] = total_epochs
        final_path = self.log_dir / 'final_metrics.json'
        with open(final_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)

        # Log final metrics to W&B
        if self.use_wandb:
            wandb_final = {f"test/{k}": v for k, v in test_metrics.items()}
            wandb_final['total_duration_seconds'] = final_metrics['experiment_duration_seconds']
            wandb.log(wandb_final)
            wandb.finish()

    def save_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """
        Save predictions to CSV.

        Args:
            predictions: List of prediction dictionaries
        """
        if not predictions:
            return

        predictions_path = self.log_dir / 'predictions.csv'
        keys = predictions[0].keys()

        with open(predictions_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(predictions)

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment config as JSON.

        Args:
            config: Configuration dictionary
        """
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def log_message(self, message: str) -> None:
        """
        Log a message to console and file.

        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        # Also save to file
        log_file = self.log_dir / 'experiment.log'
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')
