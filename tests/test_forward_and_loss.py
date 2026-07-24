import torch

from models.losses_ma_trm import MATRMLossHead
from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1


def make_batch(batch_size: int = 2, seq_len: int = 12, vocab_size: int = 12):
    return {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }


def test_forward_shapes_and_private_states(small_config: dict) -> None:
    model = MultiAgentTinyRecursiveReasoningModel_ACTV1(small_config)
    batch = make_batch()
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)

    assert outputs["logits"].shape == (2, 12, 12)
    assert outputs["agent_logits"].shape == (2, 4, 12, 12)
    assert outputs["router_mask"].shape == (2, 4)
    assert outputs["role_state_pooled"].shape == (2, 4, 64)
    assert carry.inner_carry.agent_states.shape == (2, 4, 12, 64)
    assert torch.all(outputs["active_agents"] == 2)


def test_composite_loss_backpropagates_to_router_and_links(small_config: dict) -> None:
    torch.manual_seed(3)
    base = MultiAgentTinyRecursiveReasoningModel_ACTV1(small_config)
    model = MATRMLossHead(base)
    batch = make_batch()
    carry = model.initial_carry(batch)
    _, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    loss.backward()

    assert torch.isfinite(loss)
    assert base.inner.router.proj_out.weight.grad is not None
    assert base.inner.links[0].down.weight.grad is not None
    assert base.inner.backbone.layers[0].mlp.down_proj.weight.grad is not None
    assert "consistency_loss" in metrics
    assert "mean_disagreement" in metrics
