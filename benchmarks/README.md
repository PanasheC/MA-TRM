# Benchmark Artifacts

This directory stores generated reproducibility manifests and benchmark results. The repository does not commit machine-specific outputs by default.

Run:

```bash
./scripts/run_reproducibility_suite.sh --device cuda --dtype bfloat16
```

Expected generated files include:

- `environment.json`, software, hardware, CUDA, Git, and determinism metadata.
- `parameter_budget.json`, exact TRM and MA-TRM-Lite parameter counts.
- `synthetic_comparison.json`, controlled latency, throughput, memory, and checksum measurements.
- task-specific result JSON files created by training and evaluation jobs.

Create a dataset integrity manifest before training:

```bash
python scripts/create_dataset_manifest.py data/arc1concept-aug-1000 \
  --output benchmarks/arc1_dataset_manifest.json
```

Aggregate one metric across seed result files:

```bash
python scripts/aggregate_metrics.py benchmarks/seed*.json \
  --metric metrics.exact_accuracy \
  --output benchmarks/exact_accuracy_summary.json
```

Profile supported PyTorch operator FLOPs:

```bash
python scripts/profile_flops.py --model ma-trm --device cuda \
  --output benchmarks/ma_trm_flops.json
```

Synthetic measurements validate the benchmark harness. They are not task-accuracy results.
