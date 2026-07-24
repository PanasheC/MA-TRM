from train.phase_control import configure_optimization_phase
from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1


def test_phase_control_freezes_expected_groups(small_config: dict) -> None:
    model = MultiAgentTinyRecursiveReasoningModel_ACTV1(small_config)
    report = configure_optimization_phase(model, "links")
    assert report.trainable_parameters > 0
    assert report.frozen_parameters > 0
    assert not model.inner.backbone.layers[0].mlp.down_proj.weight.requires_grad
    assert model.inner.links[0].down.weight.requires_grad
    assert model.inner.router.proj_out.weight.requires_grad

    report = configure_optimization_phase(model, "joint")
    assert report.frozen_parameters == 0
    assert all(parameter.requires_grad for parameter in model.parameters())
