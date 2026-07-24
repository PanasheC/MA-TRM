from __future__ import annotations

from typing import Sequence

from models.collaboration.base import (
    CollaborationStage,
    CollaborationTopology,
    normalize_active_roles,
)


class DistillationTopology(CollaborationTopology):
    """Teacher roles collaborate first, then a designated student role updates."""

    name = "distillation"

    def __init__(self, student_index: int = 0) -> None:
        self.student_index = int(student_index)

    def plan(
        self,
        active_roles: Sequence[int],
        num_agents: int,
        verifier_index: int,
        critic_index: int,
    ) -> tuple[CollaborationStage, ...]:
        roles = normalize_active_roles(active_roles, num_agents)
        teachers = tuple(role for role in roles if role != self.student_index)
        stages: list[CollaborationStage] = []
        if teachers:
            stages.append(CollaborationStage(teachers, "teacher_ensemble"))
        if self.student_index in roles:
            stages.append(CollaborationStage((self.student_index,), "student"))
        if not stages:
            stages.append(CollaborationStage(roles, "fallback"))
        return tuple(stages)
