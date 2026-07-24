"""Shared latent workspace and cell-level recursive attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.layers import CastedLinear, rms_norm


@dataclass
class CellAttentionDecision:
    mask: torch.Tensor
    uncertainty: torch.Tensor
    active_fraction: torch.Tensor


class SharedLatentWorkspace(nn.Module):
    """Gated residual aggregation of continuous agent messages."""

    def __init__(self, hidden_size: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.gate = CastedLinear(hidden_size, 1, bias=True)
        with torch.no_grad():
            if self.gate.bias is not None:
                self.gate.bias.fill_(-1.0)

    def forward(
        self,
        workspace: torch.Tensor,
        messages: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate ``[B, A, L, D]`` messages with ``[B, A]`` weights."""

        if messages.ndim != 4:
            raise ValueError(f"Expected messages [B,A,L,D], received {messages.shape}.")
        if weights.ndim != 2:
            raise ValueError(f"Expected weights [B,A], received {weights.shape}.")
        normalized_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aggregate = torch.einsum("ba,bald->bld", normalized_weights.to(messages.dtype), messages)
        gate = torch.sigmoid(self.gate(aggregate))
        return rms_norm(
            workspace + gate * aggregate,
            variance_epsilon=self.norm_eps,
        )


class CellRecursiveAttention(nn.Module):
    """Focus recursive updates on uncertain output cells."""

    def __init__(
        self,
        threshold: float = 0.35,
        temperature: float = 0.10,
        global_refresh_interval: int = 2,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.global_refresh_interval = int(global_refresh_interval)

    def forward(
        self,
        logits: torch.Tensor,
        prefix_len: int,
        round_index: int,
        enabled: bool = True,
    ) -> CellAttentionDecision:
        batch_size, seq_len, vocab_size = logits.shape
        if not enabled or (
            self.global_refresh_interval > 0
            and round_index % self.global_refresh_interval == 0
        ):
            mask = torch.ones(
                batch_size, seq_len + prefix_len, 1, device=logits.device, dtype=logits.dtype
            )
            uncertainty = torch.ones(
                batch_size, seq_len, device=logits.device, dtype=torch.float32
            )
            return CellAttentionDecision(
                mask=mask,
                uncertainty=uncertainty,
                active_fraction=torch.ones(batch_size, device=logits.device),
            )

        probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
        max_entropy = torch.log(
            torch.tensor(float(max(vocab_size, 2)), device=logits.device)
        )
        uncertainty = entropy / max_entropy
        if self.training:
            cell_mask = torch.sigmoid(
                (uncertainty - self.threshold) / max(self.temperature, 1e-6)
            )
        else:
            cell_mask = (uncertainty >= self.threshold).to(logits.dtype)

        if prefix_len > 0:
            prefix = torch.ones(
                batch_size, prefix_len, device=logits.device, dtype=cell_mask.dtype
            )
            cell_mask = torch.cat((prefix, cell_mask), dim=-1)
        mask = cell_mask.unsqueeze(-1).to(logits.dtype)
        active_fraction = cell_mask[:, prefix_len:].mean(dim=-1).to(torch.float32)
        return CellAttentionDecision(
            mask=mask,
            uncertainty=uncertainty,
            active_fraction=active_fraction,
        )
