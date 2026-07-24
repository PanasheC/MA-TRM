import torch

from models.routing import SparseDynamicRouter


def test_router_selects_top_k_and_mandatory_verifier() -> None:
    torch.manual_seed(0)
    router = SparseDynamicRouter(
        hidden_size=16,
        num_agents=4,
        top_k=2,
        hidden_router_size=8,
        mandatory_agents=(3,),
    )
    decision = router(torch.randn(5, 16))
    assert torch.all(decision.hard_mask.sum(dim=-1) == 2)
    assert torch.all(decision.hard_mask[:, 3] == 1)
    assert torch.allclose(decision.probabilities.sum(dim=-1), torch.ones(5))


def test_router_is_deterministic_in_evaluation() -> None:
    torch.manual_seed(11)
    router = SparseDynamicRouter(16, 4, 2, 8, mandatory_agents=(3,))
    router.eval()
    state = torch.randn(3, 16)
    first = router(state).hard_mask
    second = router(state).hard_mask
    assert torch.equal(first, second)
