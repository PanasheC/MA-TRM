"""Collaboration topology planning interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CollaborationStage:
    """Roles executed from a common workspace before one aggregation."""

    roles: tuple[int, ...]
    name: str


class CollaborationTopology(ABC):
    name: str

    @abstractmethod
    def plan(
        self,
        active_roles: Sequence[int],
        num_agents: int,
        verifier_index: int,
        critic_index: int,
    ) -> tuple[CollaborationStage, ...]:
        raise NotImplementedError


def normalize_active_roles(active_roles: Sequence[int], num_agents: int) -> tuple[int, ...]:
    roles = tuple(sorted(set(int(i) for i in active_roles)))
    if any(i < 0 or i >= num_agents for i in roles):
        raise ValueError(f"Invalid active roles {roles} for {num_agents} agents.")
    if not roles:
        raise ValueError("At least one agent must be active.")
    return roles
