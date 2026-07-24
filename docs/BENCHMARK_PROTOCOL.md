# Benchmark Protocol

## Goal

Determine whether MA-TRM improves exact structured reasoning accuracy and accuracy per unit of computation relative to TRM. Every comparison must separate architecture effects from parameter count, training compute, inference compute, augmentation, task identity embeddings, test-time voting, and random seed variance.

## Required models

1. Released TRM configuration.
2. TRM with the same trainable parameter count as MA-TRM.
3. TRM with matched training FLOPs.
4. TRM with matched inference FLOPs.
5. MA-TRM-Lite sequential topology.
6. MA-TRM mixture topology.
7. MA-TRM deliberation topology.
8. MA-TRM without learned RecursiveLinks.
9. MA-TRM with decoded categorical handoffs.
10. MA-TRM with fixed recursion depth.
11. MA-TRM without cell-level attention.
12. MA-TRM without role adapters.

## Datasets

- Sudoku-Extreme for architecture selection and seed analysis.
- Maze-Hard for spatial path reasoning.
- ARC-AGI-1 for abstraction and transformation reasoning.
- ARC-AGI-2 for harder generalization, with explicit leakage controls.

Record the source archive hash, builder command, augmentation count, split names, sequence length, vocabulary, number of puzzle identities, and resulting file hashes.

## Seed policy

Use seeds 0, 1, 2, 3, and 4 for Sudoku and Maze. Use at least seeds 0, 1, and 2 for full ARC experiments. Report mean, standard deviation, median, best run, and every individual result.

## Parameter accounting

Report separately:

- trainable `torch.nn.Parameter` elements;
- non-parameter puzzle identity buffers;
- optimizer state bytes;
- checkpoint bytes;
- active parameters per inference call when physical sparsity is enabled.

## Compute matching

Profile a complete training step and a complete evaluation trajectory. Include:

- backbone calls;
- active roles per round;
- ACT calls per puzzle;
- active cell fraction;
- forward and backward FLOPs;
- communication and aggregation FLOPs;
- wall-clock time after warm-up;
- peak allocated and reserved VRAM.

A compute-matched TRM baseline should adjust its local recursion cycles or ACT steps until measured median inference FLOPs are within 5 percent of MA-TRM.

## Accuracy and calibration

Report:

- exact puzzle accuracy;
- valid-cell accuracy;
- verifier sequence AUROC and Brier score;
- cell verifier expected calibration error;
- expected Hamming error and observed Hamming error;
- pass@1 and voting results as separate metrics;
- disagreement on correct and incorrect trajectories.

## Systems controls

- Run all timed models on the same GPU type and server.
- Record GPU clocks, power limits, driver, CUDA, cuDNN, PyTorch, NCCL, and compiler settings.
- Use at least five warm-up iterations for microbenchmarks.
- Synchronize CUDA before and after every timed region.
- Do not compare a compiled model with an uncompiled model without reporting both conditions.
- Report `physical_sparse_eval` and routing granularity.

## Reproducibility bundle

Each final run directory should include:

```text
all_config.yaml
environment.json
dataset_manifest.json
parameter_budget.json
metrics.jsonl
checkpoint
predictions
profiler_trace
stdout.log
stderr.log
```

The Git commit must be clean or the patch must be stored with the run.
