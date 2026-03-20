"""PyTorch MLP classifier trained on pre-computed VLM embeddings."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple

from src.training.metrics import MetricsComputer


# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """Wraps numpy embedding arrays as a PyTorch Dataset."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(embeddings).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# MLP model
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingMLP(nn.Module):
    """MLP classifier on top of frozen VLM embeddings."""

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(inplace=True),
                       nn.Dropout(p=dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    num_classes: int,
    task: str,
    hidden_dims: List[int],
    dropout: float = 0.3,
    lr: float = 1e-3,
    num_epochs: int = 30,
    batch_size: int = 512,
    patience: int = 7,
    device: str = "cuda:0",
    class_weights: torch.Tensor = None,
) -> Dict[str, Any]:
    """
    Train an MLP on pre-computed embeddings.

    Returns:
        dict with 'val_metrics', 'test_metrics', 'best_epoch', 'params'
    """
    train_loader = DataLoader(EmbeddingDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(EmbeddingDataset(X_val, y_val),
                              batch_size=batch_size * 4, shuffle=False)
    test_loader  = DataLoader(EmbeddingDataset(X_test, y_test),
                              batch_size=batch_size * 4, shuffle=False)

    model = EmbeddingMLP(X_train.shape[1], hidden_dims, num_classes, dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)
                                    if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss, val_preds, val_probs = _evaluate_loader(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best weights and evaluate on test
    model.load_state_dict(best_state)
    model.eval()

    _, val_preds, val_probs = _evaluate_loader(model, val_loader, None, device)
    _, test_preds, test_probs = _evaluate_loader(model, test_loader, None, device)

    val_metrics  = MetricsComputer.compute_probabilities_metrics(
        val_probs,  y_val,  num_classes, task)
    test_metrics = MetricsComputer.compute_probabilities_metrics(
        test_probs, y_test, num_classes, task)

    return {
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "best_epoch":   best_epoch + 1,
        "params": {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "lr": lr,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        },
    }


def _evaluate_loader(model, loader, criterion, device):
    """Run inference over a DataLoader. Returns (mean_loss, preds, probs)."""
    losses, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            if criterion is not None:
                losses.append(criterion(logits, y_b).item())
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, np.array(all_preds), np.array(all_probs)
