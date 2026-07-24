import torch

from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1


def build_model(config: dict, seed: int):
    torch.manual_seed(seed)
    model = MultiAgentTinyRecursiveReasoningModel_ACTV1(config)
    model.eval()
    return model


def test_same_seed_produces_same_cpu_output(small_config: dict) -> None:
    config = dict(small_config)
    config["physical_sparse_eval"] = False
    torch.manual_seed(99)
    batch = {
        "inputs": torch.randint(0, 12, (2, 12)),
        "labels": torch.randint(0, 12, (2, 12)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }
    model_a = build_model(config, 17)
    model_b = build_model(config, 17)
    with torch.inference_mode():
        carry_a = model_a.initial_carry(batch)
        carry_b = model_b.initial_carry(batch)
        _, outputs_a = model_a(carry_a, batch)
        _, outputs_b = model_b(carry_b, batch)
    assert torch.equal(outputs_a["logits"], outputs_b["logits"])
    assert torch.equal(outputs_a["router_mask"], outputs_b["router_mask"])
