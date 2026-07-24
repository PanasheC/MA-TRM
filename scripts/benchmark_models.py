#!/usr/bin/env python3
"""Synthetic, parameter-matched TRM versus MA-TRM latency benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import argparse
from dataclasses import asdict, dataclass
import json
import statistics
import time
import torch

from models.recursive_reasoning.ma_trm import MultiAgentTinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from scripts.collect_environment import collect


@dataclass
class BenchmarkResult:
    model: str
    parameters: int
    batch_size: int
    seq_len: int
    median_ms: float
    mean_ms: float
    samples_per_second: float
    peak_memory_bytes: int | None
    output_checksum: float


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(
    name: str,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    model.eval()
    durations: list[float] = []
    checksum = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index in range(warmup + iterations):
            carry = model.initial_carry(batch)
            synchronize(device)
            start = time.perf_counter()
            carry, outputs = model(carry, batch)
            synchronize(device)
            elapsed = time.perf_counter() - start
            checksum = float(outputs["logits"].to(torch.float32).sum().item())
            if index >= warmup:
                durations.append(elapsed)
    median = statistics.median(durations)
    mean = statistics.mean(durations)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return BenchmarkResult(
        model=name,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        batch_size=batch["inputs"].shape[0],
        seq_len=batch["inputs"].shape[1],
        median_ms=median * 1000.0,
        mean_ms=mean * 1000.0,
        samples_per_second=batch["inputs"].shape[0] / median,
        peak_memory_bytes=peak,
        output_checksum=checksum,
    )


def common_config(args: argparse.Namespace, dtype_name: str) -> dict:
    return {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": args.vocab_size,
        "H_cycles": args.rounds,
        "L_cycles": args.ma_local_cycles,
        "H_layers": 0,
        "L_layers": args.layers,
        "hidden_size": args.hidden_size,
        "expansion": 4,
        "num_heads": args.heads,
        "pos_encodings": "rope",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.0,
        "forward_dtype": dtype_name,
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=81)
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trm-local-cycles", type=int, default=6)
    parser.add_argument("--ma-local-cycles", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/synthetic_comparison.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    common = common_config(args, args.dtype)
    trm_config = dict(common)
    trm_config["L_cycles"] = args.trm_local_cycles
    trm = TinyRecursiveReasoningModel_ACTV1(trm_config).to(device)
    ma_config = dict(common)
    ma_config["L_cycles"] = args.ma_local_cycles
    ma_config.update(
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
    ma_trm = MultiAgentTinyRecursiveReasoningModel_ACTV1(ma_config).to(device)

    batch = {
        "inputs": torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device),
        "labels": torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device),
        "puzzle_identifiers": torch.zeros(args.batch_size, dtype=torch.long, device=device),
    }
    results = [
        timed_forward("TRM", trm, batch, device, args.warmup, args.iterations),
        timed_forward("MA-TRM-Lite", ma_trm, batch, device, args.warmup, args.iterations),
    ]
    payload = {
        "benchmark": {
            "seed": args.seed,
            "device": str(device),
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trm_local_cycles": args.trm_local_cycles,
            "ma_local_cycles": args.ma_local_cycles,
            "note": "One ACT call, synthetic inputs, no task accuracy claim.",
        },
        "environment": collect(),
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()
