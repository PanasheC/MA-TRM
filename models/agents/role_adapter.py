"""Low-rank role adapters used by the shared MA-TRM backbone."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.layers import CastedLinear


class LowRankRoleAdapter(nn.Module):
    """A residual low-rank adapter with stable near-identity initialization.

    The adapter is intentionally small. For hidden width ``d`` and rank ``r``,
    it adds ``2dr + 1`` parameters rather than a second dense backbone.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"Adapter rank must be positive, received {rank}.")
        self.hidden_size = hidden_size
        self.rank = rank
        self.down = CastedLinear(hidden_size, rank, bias=False)
        self.up = CastedLinear(rank, hidden_size, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(init_scale, 1e-8))))

        # Keep the initial system close to the TRM backbone while allowing each
        # role to specialize immediately through a small, nonzero update.

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = self.up(F.silu(self.down(hidden_states)))
        scale = self.log_scale.exp().to(update.dtype)
        return hidden_states + scale * update


class RoleAdapterBank(nn.Module):
    """One adapter for each role at each shared backbone layer."""

    def __init__(
        self,
        num_roles: int,
        num_layers: int,
        hidden_size: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.num_roles = num_roles
        self.num_layers = num_layers
        self.adapters = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        LowRankRoleAdapter(hidden_size=hidden_size, rank=rank)
                        for _ in range(num_layers)
                    ]
                )
                for _ in range(num_roles)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        role_index: int,
        layer_index: int,
    ) -> torch.Tensor:
        if not 0 <= role_index < self.num_roles:
            raise IndexError(f"Role index {role_index} is outside [0, {self.num_roles}).")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"Layer index {layer_index} is outside [0, {self.num_layers})."
            )
        return self.adapters[role_index][layer_index](hidden_states)
