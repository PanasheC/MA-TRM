from __future__ import annotations

import pytest


@pytest.fixture
def small_config() -> dict:
    return {
        "batch_size": 2,
        "seq_len": 12,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 12,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 0,
        "L_layers": 1,
        "gradient_rounds": 1,
        "hidden_size": 64,
        "expansion": 2,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 3,
        "halt_min_steps": 2,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
        "num_agents": 4,
        "role_names": ["pattern", "transform", "critic", "verifier"],
        "active_agents": 2,
        "adapter_rank": 4,
        "link_rank": 4,
        "router_hidden_size": 16,
        "physical_sparse_eval": False,
    }
