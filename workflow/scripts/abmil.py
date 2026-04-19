"""Gated attention-MIL head (Ilse et al., ICML 2018) with multi-task regression.

Input:  (B, N, D) tile embeddings, optional (B, N) bool mask.
Output: (B, n_out) predictions, (B, N) attention weights (sum to 1 per slide).
"""
from __future__ import annotations

import torch
from torch import nn


class GatedABMIL(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_out: int,
                 trunk_hidden: int = 512, dropout: float = 0.25):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.U = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(trunk_hidden, n_out)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, D)
        tanh = torch.tanh(self.V(x))
        sigm = torch.sigmoid(self.U(x))
        gated = tanh * sigm                                  # (B, N, H)
        scores = self.w(gated).squeeze(-1)                   # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)                  # (B, N)
        if mask is not None:
            attn = attn.masked_fill(~mask, 0.0)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("bn,bnd->bd", attn, x)         # (B, D)
        return self.head(self.trunk(pooled)), attn
