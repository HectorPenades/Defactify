"""Training loop implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

from src.training.base import BaseTrainer
from src.training.losses import get_loss_function, get_loss_label
from src.training.metrics import MetricsComputer
from src.utils.logging import ExperimentLogger
from src.utils.reproducibility import ReproducibilityManager


class DefaultTrainer(BaseTrainer):
    """Default trainer implementation."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict[str, Any],
                 logger: Optional[ExperimentLogger] = None):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration dictionary
            logger: Optional experiment logger
        """
        super().__init__(model, train_loader, val_loader, config)
        self.logger = logger
        self.num_classes = config.get('model', {}).get('num_classes', 6)
        self.task = config.get('task', 'multiclass')

        # Build loss function (weights and type fully driven by config)
        self.criterion = get_loss_function(config).to(self.device)
        self.loss_label = get_loss_label(config)

        # Build optimizer
        lr = float(config.get('training', {}).get('lr', 1e-3))
        weight_decay = float(config.get('training', {}).get('weight_decay', 1e-5))
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Build scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('training', {}).get('scheduler_factor', 0.5),
            patience=config.get('training', {}).get('scheduler_patience', 5),
            verbose=True
        )

        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _save_full_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float],
                               patience_counter: int) -> None:
        """Save complete training state (model + optimizer + scheduler + loop state)."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': patience_counter,
            'metrics': metrics,
        }, path)

    def _model_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass that transparently handles fusion (two-input) models."""
        images = batch['image'].to(self.device)
        if 'fft' in batch:
            return self.model(images, batch['fft'].to(self.device))
        return self.model(images)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        losses = []
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self._model_forward(batch)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Store metrics
            losses.append(loss.item())
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

            pbar.set_postfix({'loss': np.mean(losses)})

        # Compute epoch metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = (all_preds == all_labels).mean()

        return {
            'loss': np.mean(losses),
            'accuracy': float(accuracy),
            'f1_macro': float(MetricsComputer.compute_classification_metrics(
                all_preds, all_labels, self.num_classes, self.task
            )['f1_macro'])
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for batch in pbar:
                labels = batch['label'].to(self.device)

                # Forward pass
                logits = self._model_forward(batch)
                loss = self.criterion(logits, labels)

                # Store metrics
                losses.append(loss.item())
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                preds = logits.argmax(dim=1).detach().cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.detach().cpu().numpy())
                all_probs.extend(probs)

                pbar.set_postfix({'loss': np.mean(losses)})

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = MetricsComputer.compute_probabilities_metrics(
            all_probs, all_labels, self.num_classes
        )

        return {
            'loss': np.mean(losses),
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }

    def train(self, num_epochs: int = 100, patience: int = 10,
              resume_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Full training loop with early stopping.

        Args:
            num_epochs: Total number of epochs (absolute, not additional).
            patience: Early stopping patience.
            resume_state: Dict from a checkpoint produced by _save_full_checkpoint.
                          If provided, training resumes from the saved epoch.
        """
        # Restore state when resuming
        start_epoch = 0
        patience_counter = 0
        if resume_state is not None:
            start_epoch = resume_state['epoch'] + 1
            patience_counter = resume_state['patience_counter']
            self.best_val_loss = resume_state['best_val_loss']
            self.best_epoch = resume_state['best_epoch']
            self.optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            self.scheduler.load_state_dict(resume_state['scheduler_state_dict'])
            if self.logger:
                self.logger.log_message(
                    f"Resuming from epoch {start_epoch} "
                    f"(best val loss so far: {self.best_val_loss:.4f})"
                )

        for epoch in range(start_epoch, num_epochs):
            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log
            if self.logger:
                self.logger.log_epoch(epoch, train_metrics, val_metrics)
                self.logger.log_message(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val BalAcc: {val_metrics['balanced_accuracy']:.4f} | "
                    f"Val F1-macro: {val_metrics['f1_macro']:.4f}"
                )

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint (full state so it can also be used to resume)
                if self.logger:
                    best_path = self.logger.log_dir / 'checkpoint_best.pth'
                    self._save_full_checkpoint(str(best_path), epoch, val_metrics, patience_counter)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.logger:
                        self.logger.log_message(
                            f"Early stopping at epoch {epoch+1} (best epoch: {self.best_epoch+1})"
                        )
                    break

            # Save last checkpoint every epoch (enables resume from interruption)
            if self.logger:
                last_path = self.logger.log_dir / 'checkpoint_last.pth'
                self._save_full_checkpoint(str(last_path), epoch, val_metrics, patience_counter)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

        if self.logger:
            self.logger.save_metrics()
