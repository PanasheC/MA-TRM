#!/usr/bin/env python3
"""Create a deterministic SHA-256 manifest for a generated benchmark dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    entries = []
    total_bytes = 0
    for path in sorted(p for p in dataset_dir.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(dataset_dir).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    return {
        "dataset_directory": str(dataset_dir),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"{manifest['file_count']} files, {manifest['total_bytes']} bytes, "
        f"manifest {manifest['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
