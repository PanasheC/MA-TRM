from __future__ import annotations

from typing import Sequence

from models.collaboration.base import (
    CollaborationStage,
    CollaborationTopology,
    normalize_active_roles,
)


class DeliberationTopology(CollaborationTopology):
    name = "deliberation"

    def plan(
        self,
        active_roles: Sequence[int],
        num_agents: int,
        verifier_index: int,
        critic_index: int,
    ) -> tuple[CollaborationStage, ...]:
        roles = normalize_active_roles(active_roles, num_agents)
        proposals = tuple(
            role for role in roles if role not in {critic_index, verifier_index}
        )
        stages: list[CollaborationStage] = []
        if proposals:
            stages.append(CollaborationStage(proposals, "proposals"))
        if critic_index in roles:
            stages.append(CollaborationStage((critic_index,), "critique"))
        if verifier_index in roles:
            stages.append(CollaborationStage((verifier_index,), "verification"))
        if not stages:
            stages.append(CollaborationStage(roles, "fallback"))
        return tuple(stages)
