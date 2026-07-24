from models.collaboration.base import CollaborationStage, CollaborationTopology
from models.collaboration.sequential import SequentialTopology
from models.collaboration.mixture import MixtureTopology
from models.collaboration.deliberation import DeliberationTopology
from models.collaboration.distillation import DistillationTopology


def build_topology(name: str, student_index: int = 0) -> CollaborationTopology:
    normalized = name.strip().lower()
    if normalized == "sequential":
        return SequentialTopology()
    if normalized == "mixture":
        return MixtureTopology()
    if normalized == "deliberation":
        return DeliberationTopology()
    if normalized == "distillation":
        return DistillationTopology(student_index=student_index)
    raise ValueError(
        f"Unknown collaboration topology '{name}'. Expected sequential, mixture, "
        "deliberation, or distillation."
    )


__all__ = [
    "CollaborationStage",
    "CollaborationTopology",
    "SequentialTopology",
    "MixtureTopology",
    "DeliberationTopology",
    "DistillationTopology",
    "build_topology",
]
