#!/usr/bin/env python3
"""Report a reproducible TRM versus MA-TRM parameter budget."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import argparse
import json
from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def base_config(hidden_size: int, vocab_size: int, seq_len: int) -> dict:
    return {
        "batch_size": 2,
        "seq_len": seq_len,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": vocab_size,
        "H_cycles": 3,
        "L_cycles": 2,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": hidden_size,
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


def count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def ma_breakdown(model) -> dict[str, int]:
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("inner.backbone.layers"):
            group = "shared_backbone"
        elif name.startswith("inner.backbone.adapters"):
            group = "role_adapters"
        elif name.startswith("inner.links"):
            group = "recursive_links"
        elif name.startswith("inner.router"):
            group = "router"
        elif name.startswith("inner.workspace_update"):
            group = "workspace"
        elif name.startswith("inner.role_embeddings"):
            group = "role_embeddings"
        elif name.startswith("inner.embed_tokens") or name.startswith("inner.lm_head"):
            group = "io_embeddings_and_answer_head"
        elif name.startswith("inner.cell_verifier_head") or name.startswith("inner.q_head"):
            group = "verification_and_halting"
        else:
            group = "other"
        groups[group] = groups.get(group, 0) + parameter.numel()
    return dict(sorted(groups.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=81)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    trm_cfg = base_config(args.hidden_size, args.vocab_size, args.seq_len)
    trm_cfg["L_cycles"] = 6
    ma_cfg = base_config(args.hidden_size, args.vocab_size, args.seq_len)
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
            "physical_sparse_eval": True,
        }
    )

    trm = TinyRecursiveReasoningModel_ACTV1(trm_cfg)
    ma_trm = MultiAgentTinyRecursiveReasoningModel_ACTV1(ma_cfg)
    trm_count = count(trm)
    ma_count = count(ma_trm)
    result = {
        "TRM_parameters": trm_count,
        "MA_TRM_Lite_parameters": ma_count,
        "absolute_overhead": ma_count - trm_count,
        "relative_overhead_percent": 100.0 * (ma_count - trm_count) / trm_count,
        "within_7M_target": 6_500_000 <= ma_count <= 7_500_000,
        "MA_TRM_Lite_breakdown": ma_breakdown(ma_trm),
        "counting_rule": "torch parameters only, excluding puzzle identity buffers",
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
