from __future__ import annotations

from typing import Sequence

from models.collaboration.base import (
    CollaborationStage,
    CollaborationTopology,
    normalize_active_roles,
)


class MixtureTopology(CollaborationTopology):
    name = "mixture"

    def plan(
        self,
        active_roles: Sequence[int],
        num_agents: int,
        verifier_index: int,
        critic_index: int,
    ) -> tuple[CollaborationStage, ...]:
        roles = normalize_active_roles(active_roles, num_agents)
        return (CollaborationStage(roles, "parallel_mixture"),)
