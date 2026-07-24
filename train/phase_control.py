"""Parameter-freezing policies for MA-TRM inner and outer optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn


@dataclass(frozen=True)
class PhaseReport:
    phase: str
    trainable_parameters: int
    frozen_parameters: int
    trainable_tensors: int
    frozen_tensors: int


def _unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    # Loss head, torch.compile, and similar wrappers expose one of these names.
    for _ in range(4):
        if hasattr(current, "_orig_mod"):
            current = getattr(current, "_orig_mod")
            continue
        if hasattr(current, "model"):
            current = getattr(current, "model")
            continue
        break
    return current


def _set_named_parameters(
    module: nn.Module,
    trainable_prefixes: Iterable[str] | None,
) -> PhaseReport:
    prefixes = None if trainable_prefixes is None else tuple(trainable_prefixes)
    trainable_parameters = 0
    frozen_parameters = 0
    trainable_tensors = 0
    frozen_tensors = 0
    for name, parameter in module.named_parameters():
        enabled = prefixes is None or any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable_parameters += parameter.numel()
            trainable_tensors += 1
        else:
            frozen_parameters += parameter.numel()
            frozen_tensors += 1
    return PhaseReport(
        phase="",
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
        trainable_tensors=trainable_tensors,
        frozen_tensors=frozen_tensors,
    )


def configure_optimization_phase(model: nn.Module, phase: str = "joint") -> PhaseReport:
    """Apply the paper's inner, link, and joint optimization phases.

    ``agents`` trains the shared backbone, role adapters, embeddings, and task
    heads. ``links`` freezes those components and trains RecursiveLinks, routing,
    workspace integration, and halting. ``joint`` enables every parameter.
    """

    phase = phase.strip().lower()
    base = _unwrap_model(model)
    if base.__class__.__name__ not in {
        "MultiAgentTinyRecursiveReasoningModel_ACTV1",
        "MATRMLite",
        "MA_TRM_Lite",
    }:
        if phase != "joint":
            raise ValueError(
                f"optimization_phase='{phase}' is only supported by MA-TRM, "
                f"received {base.__class__.__name__}."
            )
        report = _set_named_parameters(base, None)
        return PhaseReport(phase=phase, **{k: v for k, v in report.__dict__.items() if k != "phase"})

    if phase == "joint":
        prefixes = None
    elif phase == "agents":
        prefixes = (
            "inner.embed_tokens",
            "inner.puzzle_emb",
            "inner.backbone",
            "inner.role_embeddings",
            "inner.lm_head",
            "inner.cell_verifier_head",
            "inner.q_head",
        )
    elif phase == "links":
        prefixes = (
            "inner.links",
            "inner.router",
            "inner.workspace_update",
            "inner.cell_verifier_head",
            "inner.q_head",
        )
    else:
        raise ValueError("phase must be one of: agents, links, joint.")

    report = _set_named_parameters(base, prefixes)
    return PhaseReport(phase=phase, **{k: v for k, v in report.__dict__.items() if k != "phase"})
