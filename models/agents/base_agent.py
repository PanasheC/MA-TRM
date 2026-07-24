"""Core abstractions for role-conditioned tiny reasoning agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


DEFAULT_AGENT_ROLES: tuple[str, ...] = (
    "pattern",
    "transform",
    "critic",
    "verifier",
)


@dataclass(frozen=True)
class AgentRoleSpec:
    """Human-readable description of one MA-TRM role."""

    name: str
    description: str


DEFAULT_ROLE_SPECS: tuple[AgentRoleSpec, ...] = (
    AgentRoleSpec("pattern", "Detects objects, repetition, symmetry, and relations."),
    AgentRoleSpec("transform", "Proposes spatial and symbolic state transformations."),
    AgentRoleSpec("critic", "Finds contradictions and unsupported transformations."),
    AgentRoleSpec("verifier", "Estimates cell and sequence correctness and controls recursion."),
)


@dataclass
class AgentUpdate:
    """Result of one role-conditioned agent update."""

    state: torch.Tensor
    message: torch.Tensor
    role_index: int


def validate_role_names(role_names: Sequence[str], num_agents: int) -> tuple[str, ...]:
    """Validate and normalize configured role names."""

    names = tuple(str(name).strip().lower() for name in role_names)
    if len(names) != num_agents:
        raise ValueError(
            f"Expected {num_agents} role names, received {len(names)}: {names}."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Agent role names must be unique, received {names}.")
    return names
