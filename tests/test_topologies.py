import pytest

from models.collaboration import build_topology


@pytest.mark.parametrize(
    "name,expected_stage_names",
    [
        ("sequential", ("role_0", "role_1", "role_3")),
        ("mixture", ("parallel_mixture",)),
        ("deliberation", ("proposals", "verification")),
        ("distillation", ("teacher_ensemble", "student")),
    ],
)
def test_topology_plans(name: str, expected_stage_names: tuple[str, ...]) -> None:
    topology = build_topology(name, student_index=0)
    plan = topology.plan(
        active_roles=(0, 1, 3),
        num_agents=4,
        verifier_index=3,
        critic_index=2,
    )
    assert tuple(stage.name for stage in plan) == expected_stage_names
