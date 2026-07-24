#!/usr/bin/env python3
"""Profile one ACT call and report operator FLOPs for TRM or MA-TRM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def config(args: argparse.Namespace) -> dict:
    return {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": args.vocab_size,
        "H_cycles": args.rounds,
        "L_cycles": args.local_cycles,
        "H_layers": 0,
        "L_layers": args.layers,
        "hidden_size": args.hidden_size,
        "expansion": 4,
        "num_heads": args.heads,
        "pos_encodings": "rope",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.0,
        "forward_dtype": args.dtype,
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("trm", "ma-trm"), default="ma-trm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=81)
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-cycles", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/flops.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    model_config = config(args)
    if args.model == "trm":
        model = TinyRecursiveReasoningModel_ACTV1(model_config)
    else:
        model_config.update(
            {
                "gradient_rounds": 1,
                "halt_min_steps": 2,
                "adaptive_eval": False,
                "num_agents": 4,
                "role_names": ["pattern", "transform", "critic", "verifier"],
                "active_agents": 2,
                "adapter_rank": 8,
                "link_rank": 8,
                "router_hidden_size": 64,
                "physical_sparse_eval": True,
            }
        )
        model = MultiAgentTinyRecursiveReasoningModel_ACTV1(model_config)
    model = model.to(device).eval()
    batch = {
        "inputs": torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device),
        "labels": torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device),
        "puzzle_identifiers": torch.zeros(args.batch_size, dtype=torch.long, device=device),
    }
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch.inference_mode(), profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        carry = model.initial_carry(batch)
        model(carry, batch)
    events = prof.key_averages()
    known_flops = sum(int(event.flops or 0) for event in events)
    top_events = sorted(events, key=lambda event: int(event.flops or 0), reverse=True)[:20]
    payload = {
        "model": args.model,
        "known_operator_flops": known_flops,
        "warning": "PyTorch profiler FLOPs cover supported operators only.",
        "top_events": [
            {
                "key": event.key,
                "flops": int(event.flops or 0),
                "cpu_time_total_us": float(event.cpu_time_total),
                "cuda_time_total_us": float(getattr(event, "device_time_total", 0.0)),
            }
            for event in top_events
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
