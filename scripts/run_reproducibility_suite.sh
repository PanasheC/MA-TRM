#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

python scripts/collect_environment.py --output benchmarks/environment.json
python scripts/count_parameters.py --output benchmarks/parameter_budget.json
python scripts/smoke_test.py --topology sequential
python scripts/smoke_test.py --topology mixture
python scripts/smoke_test.py --topology deliberation
python scripts/smoke_test.py --topology distillation
pytest -q
python scripts/benchmark_models.py "$@"
