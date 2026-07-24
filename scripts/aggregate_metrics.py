#!/usr/bin/env python3
"""Aggregate one numeric metric across reproducible JSON run manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


def nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--metric", required=True, help="Dotted JSON path")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = []
    for path in args.files:
        payload = json.loads(path.read_text())
        value = float(nested_value(payload, args.metric))
        if not math.isfinite(value):
            raise ValueError(f"Metric is not finite in {path}: {value}")
        records.append({"file": str(path), "value": value})
    values = [record["value"] for record in records]
    summary = {
        "metric": args.metric,
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "runs": records,
    }
    text = json.dumps(summary, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
