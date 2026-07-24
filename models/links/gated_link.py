"""Dense gated link retained for ablation experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.layers import CastedLinear, rms_norm
from models.links.recursive_link import RecursiveLink


class GatedRecursiveLink(RecursiveLink):
    """A full-rank residual latent link for controlled ablations."""

    def __init__(self, hidden_size: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.gate = CastedLinear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        return hidden_states + torch.sigmoid(self.gate(normalized)) * F.silu(
            self.proj(normalized)
        )
