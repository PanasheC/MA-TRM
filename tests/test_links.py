import torch

from models.links import LowRankRecursiveLink


def test_low_rank_link_preserves_shape_and_has_residual_path() -> None:
    torch.manual_seed(1)
    link = LowRankRecursiveLink(hidden_size=32, rank=4)
    hidden = torch.randn(2, 7, 32, requires_grad=True)
    output = link(hidden)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert hidden.grad is not None
    assert link.down.weight.grad is not None
