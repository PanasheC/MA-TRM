"""Disagreement, verification, and adaptive recursion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DisagreementResult:
    sequence: torch.Tensor
    cells: torch.Tensor


def multi_agent_js_divergence(
    agent_logits: torch.Tensor,
    active_mask: torch.Tensor,
    valid_cells: torch.Tensor | None = None,
) -> DisagreementResult:
    """Compute weighted Jensen-Shannon disagreement across active agents.

    Args:
        agent_logits: Tensor with shape ``[B, M, L, C]``.
        active_mask: Tensor with shape ``[B, M]``.
        valid_cells: Optional boolean tensor with shape ``[B, L]``.
    """

    if agent_logits.ndim != 4:
        raise ValueError(
            f"agent_logits must have shape [B,M,L,C], received {agent_logits.shape}."
        )
    weights = active_mask.to(torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    probs = torch.softmax(agent_logits.to(torch.float32), dim=-1)
    mixture = torch.einsum("bm,bmlc->blc", weights, probs).clamp_min(1e-9)
    log_mixture = mixture.log().unsqueeze(1)
    kl = (probs * (probs.clamp_min(1e-9).log() - log_mixture)).sum(dim=-1)
    cell_jsd = torch.einsum("bm,bml->bl", weights, kl)
    if valid_cells is None:
        sequence_jsd = cell_jsd.mean(dim=-1)
    else:
        valid = valid_cells.to(cell_jsd.dtype)
        sequence_jsd = (cell_jsd * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
    return DisagreementResult(sequence=sequence_jsd, cells=cell_jsd)


def answer_change_fraction(
    previous_logits: torch.Tensor,
    current_logits: torch.Tensor,
    previous_valid: torch.Tensor,
) -> torch.Tensor:
    """Fraction of cells whose categorical prediction changed."""

    previous = previous_logits.argmax(dim=-1)
    current = current_logits.argmax(dim=-1)
    change = (previous != current).to(torch.float32).mean(dim=-1)
    return torch.where(previous_valid, change, torch.ones_like(change))


def mean_predictive_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    return 1.0 - probabilities.max(dim=-1).values.mean(dim=-1)


def calibrated_expected_hamming_error(cell_confidence: torch.Tensor) -> torch.Tensor:
    r"""Return ``sum_q (1 - c_q)`` under the paper's calibration assumption."""

    return (1.0 - cell_confidence.to(torch.float32)).sum(dim=-1)
