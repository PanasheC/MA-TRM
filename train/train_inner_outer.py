"""Launch the three-stage MA-TRM inner and outer co-optimization schedule."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--project-name", default="MA-TRM-Reproducibility")
    parser.add_argument("--run-prefix", default="ma_trm_lite")
    parser.add_argument("--agents-epochs", type=int, default=10000)
    parser.add_argument("--links-epochs", type=int, default=5000)
    parser.add_argument("--joint-epochs", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--global-batch-size", type=int, default=768)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    env = os.environ.copy()
    env.setdefault("DISABLE_COMPILE", "1")
    launcher = [sys.executable]
    if args.nproc_per_node > 1:
        launcher = [
            "torchrun",
            f"--nproc-per-node={args.nproc_per_node}",
        ]

    phases = (
        ("agents", args.agents_epochs),
        ("links", args.links_epochs),
        ("joint", args.joint_epochs),
    )
    previous_checkpoint: str | None = None
    for phase, epochs in phases:
        run_name = f"{args.run_prefix}_{phase}_seed{args.seed}"
        checkpoint_dir = root / "checkpoints" / args.project_name / run_name
        command = launcher + [
            str(root / "pretrain.py"),
            "arch=ma_trm_lite",
            f"data_paths=[{args.data_path}]",
            f"epochs={epochs}",
            f"eval_interval={args.eval_interval}",
            f"global_batch_size={args.global_batch_size}",
            f"project_name={args.project_name}",
            f"run_name={run_name}",
            f"checkpoint_path={checkpoint_dir}",
            f"seed={args.seed}",
            f"optimization_phase={phase}",
        ]
        if previous_checkpoint is not None:
            command.append(f"load_checkpoint={previous_checkpoint}")
        command.extend(args.overrides)
        run(command, env=env)

        candidates = sorted(checkpoint_dir.glob("step_*"), key=lambda path: int(path.name.split("_")[-1]))
        if not candidates:
            raise RuntimeError(f"No checkpoint was created in {checkpoint_dir}.")
        previous_checkpoint = str(candidates[-1])


if __name__ == "__main__":
    main()
