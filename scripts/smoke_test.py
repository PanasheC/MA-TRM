#!/usr/bin/env python3
"""Run a small CPU or GPU forward and backward validation."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import argparse
import json

import torch

from models.losses_ma_trm import MATRMLossHead
from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--topology", choices=("sequential", "mixture", "deliberation", "distillation"), default="sequential")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(7)
    config = {
        "batch_size": 2,
        "seq_len": 16,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 12,
        "H_cycles": 2,
        "L_cycles": 1,
        "H_layers": 0,
        "L_layers": 2,
        "gradient_rounds": 1,
        "hidden_size": 128,
        "expansion": 2,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 4,
        "halt_min_steps": 2,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
        "num_agents": 4,
        "role_names": ["pattern", "transform", "critic", "verifier"],
        "active_agents": 3 if args.topology in {"deliberation", "distillation"} else 2,
        "adapter_rank": 4,
        "link_rank": 4,
        "router_hidden_size": 16,
        "topology": args.topology,
        "physical_sparse_eval": False,
    }
    base = MultiAgentTinyRecursiveReasoningModel_ACTV1(config).to(device)
    model = MATRMLossHead(base).to(device)
    batch = {
        "inputs": torch.randint(0, 12, (2, 16), device=device),
        "labels": torch.randint(0, 12, (2, 16), device=device),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long, device=device),
    }
    carry = model.initial_carry(batch)
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    loss.backward()
    result = {
        "topology": args.topology,
        "loss": float(loss.detach().to(torch.float32).item()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_with_gradients": sum(parameter.grad is not None for parameter in model.parameters()),
        "metrics": sorted(metrics),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
