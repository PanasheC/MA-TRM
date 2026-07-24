"""Low-rank residual RecursiveLink implementation."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.layers import CastedLinear, rms_norm
from models.links.recursive_link import RecursiveLink


class LowRankRecursiveLink(RecursiveLink):
    r"""Transform a continuous latent message without decoding it.

    The link implements

    .. math::
        R(h) = h + \sigma(g(h)) W_{up}\,\mathrm{SiLU}(W_{down}\,\mathrm{RMSNorm}(h)).
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        norm_eps: float = 1e-5,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"Link rank must be positive, received {rank}.")
        self.hidden_size = hidden_size
        self.rank = rank
        self.norm_eps = norm_eps
        self.down = CastedLinear(hidden_size, rank, bias=False)
        self.up = CastedLinear(rank, hidden_size, bias=False)
        self.gate = CastedLinear(hidden_size, 1, bias=True)
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(init_scale, 1e-8))))

        with torch.no_grad():
            if self.gate.bias is not None:
                self.gate.bias.fill_(-2.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        update = self.up(F.silu(self.down(normalized)))
        gate = torch.sigmoid(self.gate(normalized))
        scale = self.log_scale.exp().to(update.dtype)
        return hidden_states + scale * gate * update
