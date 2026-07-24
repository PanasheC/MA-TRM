import torch

from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1


def test_evaluation_physically_executes_only_top_k_roles(small_config: dict) -> None:
    config = dict(small_config)
    config["physical_sparse_eval"] = True
    model = MultiAgentTinyRecursiveReasoningModel_ACTV1(config)
    model.eval()
    calls = 0

    def hook(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.inner.backbone.register_forward_hook(hook)
    batch = {
        "inputs": torch.randint(0, 12, (2, 12)),
        "labels": torch.randint(0, 12, (2, 12)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        _, outputs = model(carry, batch)
    handle.remove()
    assert calls == config["active_agents"] * config["H_cycles"] * config["L_cycles"]
    assert torch.equal(outputs["router_mask"][0], outputs["router_mask"][1])
    assert torch.all(outputs["router_mask"].sum(dim=-1) == config["active_agents"])
