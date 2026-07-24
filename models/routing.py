"""Sparse dynamic routing for MA-TRM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from models.layers import CastedLinear


@dataclass
class RoutingDecision:
    """Router outputs used by the collaboration engine and loss head."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    hard_mask: torch.Tensor
    straight_through_mask: torch.Tensor


class SparseDynamicRouter(nn.Module):
    """Select a small role subset from the pooled shared workspace.

    Training uses a straight-through top-k mask. Inference can use the hard mask
    directly. Mandatory roles, normally the verifier, are always selected.
    """

    def __init__(
        self,
        hidden_size: int,
        num_agents: int,
        top_k: int,
        hidden_router_size: int = 64,
        mandatory_agents: Iterable[int] = (3,),
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_agents < 1:
            raise ValueError("num_agents must be at least one.")
        mandatory = tuple(sorted(set(int(i) for i in mandatory_agents)))
        if any(i < 0 or i >= num_agents for i in mandatory):
            raise ValueError(
                f"Mandatory agent indices {mandatory} are invalid for {num_agents} agents."
            )
        if top_k < len(mandatory) or top_k > num_agents:
            raise ValueError(
                f"top_k={top_k} must be between {len(mandatory)} and {num_agents}."
            )
        self.num_agents = num_agents
        self.top_k = top_k
        self.mandatory_agents = mandatory
        self.temperature = float(temperature)
        self.proj_in = CastedLinear(hidden_size, hidden_router_size, bias=True)
        self.proj_out = CastedLinear(hidden_router_size, num_agents, bias=True)

    def _hard_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[0]
        hard = torch.zeros_like(logits)
        if self.mandatory_agents:
            hard[:, list(self.mandatory_agents)] = 1.0

        remaining = self.top_k - len(self.mandatory_agents)
        if remaining > 0:
            candidate_logits = logits.clone()
            if self.mandatory_agents:
                candidate_logits[:, list(self.mandatory_agents)] = torch.finfo(
                    candidate_logits.dtype
                ).min
            # A tiny deterministic index offset makes ties stable across devices.
            tie_break = torch.arange(
                self.num_agents, device=logits.device, dtype=logits.dtype
            )
            candidate_logits = candidate_logits - tie_break.unsqueeze(0) * 1e-7
            indices = torch.topk(candidate_logits, k=remaining, dim=-1).indices
            hard.scatter_(1, indices, 1.0)
        return hard

    def forward(self, pooled_workspace: torch.Tensor) -> RoutingDecision:
        logits = self.proj_out(F.silu(self.proj_in(pooled_workspace))).to(torch.float32)
        probabilities = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        hard_mask = self._hard_top_k(logits)
        if self.training:
            straight_through = hard_mask + probabilities - probabilities.detach()
        else:
            straight_through = hard_mask
        return RoutingDecision(
            logits=logits,
            probabilities=probabilities,
            hard_mask=hard_mask,
            straight_through_mask=straight_through,
        )

    @torch.no_grad()
    def batch_active_agents(self, decision: RoutingDecision) -> list[int]:
        """Return a single physically sparse role set for a complete batch."""

        mean_logits = decision.logits.mean(dim=0, keepdim=True)
        mask = self._hard_top_k(mean_logits)[0]
        return torch.nonzero(mask > 0, as_tuple=False).flatten().cpu().tolist()
