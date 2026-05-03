"""Multi-task regression loss and label transforms for CX prediction."""
from __future__ import annotations

import torch
from torch import nn


class MultiTaskHuber(nn.Module):
    """Huber loss averaged over signatures with a learnable per-task scale.

    If task_scale is kept at 1 (default), this is a plain multi-task Huber.
    Set task_scale via `.task_scale.data = ...` from empirical per-signature
    variance to equalise the effective gradient.
    """
    def __init__(self, n_out: int, delta: float = 1.0):
        super().__init__()
        self.delta = delta
        self.register_parameter("task_scale",
                                nn.Parameter(torch.ones(n_out), requires_grad=False))
        self.register_buffer("active_mask", torch.ones(n_out))

    def set_active_mask(self, mask: torch.Tensor) -> None:
        self.active_mask = mask.to(self.active_mask.device).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per = torch.nn.functional.huber_loss(pred, target, delta=self.delta,
                                             reduction="none")
        per = per / self.task_scale.clamp_min(1e-6)
        per = per * self.active_mask
        denom = self.active_mask.sum().clamp_min(1.0) * pred.shape[0]
        return per.sum() / denom


def fit_label_transform(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """log1p + z-score stats. Apply to y before training, invert after."""
    y1 = torch.log1p(y.clamp_min(0))
    mu = y1.mean(dim=0)
    sd = y1.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mu, sd


def apply_label_transform(y: torch.Tensor, mu: torch.Tensor,
                          sd: torch.Tensor) -> torch.Tensor:
    return (torch.log1p(y.clamp_min(0)) - mu) / sd


def invert_label_transform(z: torch.Tensor, mu: torch.Tensor,
                           sd: torch.Tensor) -> torch.Tensor:
    return torch.expm1(z * sd + mu)
