"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples, focuses on hard ones.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=0  →  identical to CrossEntropyLoss
    gamma=2  →  standard setting from Lin et al. (RetinaNet, 2017)

    Optionally accepts class weights (same semantic as nn.CrossEntropyLoss).
    """

    def __init__(self, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # per-sample CE, shape (N,)
        ce = F.cross_entropy(logits, labels,
                             weight=self.weight,   # type: ignore[arg-type]
                             reduction='none')
        pt = torch.exp(-ce)                        # P(correct class)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Build loss module from config.

    Config keys (under 'loss:'):
        type             : 'ce' | 'weighted_ce' | 'focal' | 'label_smoothing'
                           (default: 'ce')
        handle_imbalance : 'weighted'  →  applies class_weights (binary only)
        class_weights    : [w0, w1]    →  default [5.0, 1.0] for 1:5 binary
        gamma            : float       →  focal focusing param  (default 2.0)
        label_smoothing  : float       →  smoothing epsilon     (default 0.1)

    The returned module must be moved to device with .to(device).
    """
    loss_cfg  = config.get('loss', {})
    task      = config.get('task', 'multiclass')
    loss_type = loss_cfg.get('type', 'ce')

    # Class weights — only for binary imbalance handling
    weights = None
    if loss_cfg.get('handle_imbalance') == 'weighted' and task == 'binary':
        w = loss_cfg.get('class_weights', [5.0, 1.0])
        weights = torch.tensor(w, dtype=torch.float32)

    if loss_type in ('ce', 'crossentropy', 'weighted_ce'):
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_type == 'focal':
        gamma = float(loss_cfg.get('gamma', 2.0))
        return FocalLoss(gamma=gamma, weight=weights)

    elif loss_type == 'label_smoothing':
        eps = float(loss_cfg.get('label_smoothing', 0.1))
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=eps)

    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            "Supported: ce, weighted_ce, focal, label_smoothing"
        )


def get_loss_label(config: Dict[str, Any]) -> str:
    """Return a concise description of the loss for logs and result tables.

    Examples:
        'ce'
        'weighted_ce(5,1)'
        'focal(gamma=2)'
        'focal(gamma=2,w=5,1)'
        'ce_ls(eps=0.1)'
    """
    loss_cfg  = config.get('loss', {})
    task      = config.get('task', 'multiclass')
    loss_type = loss_cfg.get('type', 'ce')

    has_weights = (loss_cfg.get('handle_imbalance') == 'weighted'
                   and task == 'binary')

    def _weight_str() -> str:
        w = loss_cfg.get('class_weights', [5.0, 1.0])
        return ','.join(str(int(x)) if float(x) == int(x) else str(x) for x in w)

    if loss_type == 'focal':
        gamma = loss_cfg.get('gamma', 2.0)
        base = f"focal(gamma={gamma}"
        return base + f",w={_weight_str()})" if has_weights else base + ")"

    if loss_type == 'label_smoothing':
        eps = loss_cfg.get('label_smoothing', 0.1)
        base = f"ce_ls(eps={eps}"
        return base + f",w={_weight_str()})" if has_weights else base + ")"

    # ce / weighted_ce / crossentropy
    if has_weights:
        return f"weighted_ce({_weight_str()})"
    return 'ce'
