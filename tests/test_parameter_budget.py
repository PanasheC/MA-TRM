from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def test_default_ma_trm_is_about_seven_million_parameters() -> None:
    common = {
        "batch_size": 2,
        "seq_len": 81,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 12,
        "H_cycles": 3,
        "L_cycles": 2,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
    }
    baseline_cfg = dict(common)
    baseline_cfg["L_cycles"] = 6
    ma_cfg = dict(common)
    ma_cfg.update(
        {
            "gradient_rounds": 1,
            "halt_min_steps": 2,
            "num_agents": 4,
            "role_names": ["pattern", "transform", "critic", "verifier"],
            "active_agents": 2,
            "adapter_rank": 8,
            "link_rank": 8,
            "router_hidden_size": 64,
        }
    )
    trm = TinyRecursiveReasoningModel_ACTV1(baseline_cfg)
    ma_trm = MultiAgentTinyRecursiveReasoningModel_ACTV1(ma_cfg)
    trm_parameters = sum(parameter.numel() for parameter in trm.parameters())
    ma_parameters = sum(parameter.numel() for parameter in ma_trm.parameters())

    assert 6_500_000 <= ma_parameters <= 7_500_000
    assert ma_parameters == 6_965_598
    assert (ma_parameters - trm_parameters) / trm_parameters < 0.03
