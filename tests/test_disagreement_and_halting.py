import torch

from models.halting import (
    answer_change_fraction,
    calibrated_expected_hamming_error,
    multi_agent_js_divergence,
)


def test_disagreement_zero_for_identical_agents() -> None:
    logits = torch.randn(2, 1, 5, 3).repeat(1, 4, 1, 1)
    active = torch.ones(2, 4)
    result = multi_agent_js_divergence(logits, active)
    assert torch.allclose(result.sequence, torch.zeros_like(result.sequence), atol=1e-6)


def test_answer_change_respects_previous_valid() -> None:
    previous = torch.zeros(2, 4, 3)
    current = torch.zeros(2, 4, 3)
    current[0, :, 1] = 1
    valid = torch.tensor([True, False])
    change = answer_change_fraction(previous, current, valid)
    assert change[0] == 1
    assert change[1] == 1


def test_expected_hamming_error_equation() -> None:
    confidence = torch.tensor([[0.9, 0.8, 0.5]])
    error = calibrated_expected_hamming_error(confidence)
    assert torch.allclose(error, torch.tensor([0.8]))


def test_batch_level_eval_halting_is_synchronous(small_config: dict) -> None:
    from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1

    config = dict(small_config)
    config.update({
        "halt_min_steps": 1,
        "halt_threshold": -10.0,
        "disagreement_halt_threshold": 10.0,
        "stability_halt_threshold": 1.0,
        "adaptive_eval": True,
        "eval_halt_granularity": "batch",
        "physical_sparse_eval": False,
    })
    model = MultiAgentTinyRecursiveReasoningModel_ACTV1(config)
    model.eval()
    batch = {
        "inputs": torch.randint(0, 12, (2, 12)),
        "labels": torch.randint(0, 12, (2, 12)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry, _ = model(carry, batch)
    assert torch.equal(carry.halted, torch.tensor([True, True]))


def test_sample_level_eval_preserves_completed_samples(small_config: dict) -> None:
    from types import MethodType
    from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1

    config = dict(small_config)
    config.update({
        "halt_min_steps": 1,
        "halt_threshold": 0.0,
        "disagreement_halt_threshold": 10.0,
        "stability_halt_threshold": 1.0,
        "adaptive_eval": True,
        "eval_halt_granularity": "sample",
        "physical_sparse_eval": False,
    })
    model = MultiAgentTinyRecursiveReasoningModel_ACTV1(config)
    model.eval()
    original_forward = model.inner.forward
    calls = {"count": 0}

    def controlled_forward(self, carry, batch):
        new_carry, logits, (_, q_continue), aux = original_forward(carry, batch)
        if calls["count"] == 0:
            q_halt = torch.tensor([10.0, -10.0], dtype=torch.float32)
        else:
            q_halt = torch.tensor([10.0, 10.0], dtype=torch.float32)
        calls["count"] += 1
        return new_carry, logits, (q_halt, q_continue), aux

    model.inner.forward = MethodType(controlled_forward, model.inner)
    batch = {
        "inputs": torch.randint(0, 12, (2, 12)),
        "labels": torch.randint(0, 12, (2, 12)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry, first_outputs = model(carry, batch)
        first_completed_logits = first_outputs["logits"][0].clone()
        assert torch.equal(carry.halted, torch.tensor([True, False]))
        carry, second_outputs = model(carry, batch)
    assert torch.equal(carry.halted, torch.tensor([True, True]))
    assert torch.equal(second_outputs["logits"][0], first_completed_logits)
