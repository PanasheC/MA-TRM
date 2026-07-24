# Validation Report

## Scope

This report records code-level validation completed before packaging the repository. It does not contain trained benchmark accuracy and does not claim that MA-TRM outperforms TRM.

## Parameter validation

The full 512-wide MA-TRM-Lite configuration was instantiated with vocabulary size 12 and puzzle identity embeddings disabled for core-network accounting.

| Model | Trainable parameters |
|---|---:|
| TRM baseline | 6,829,058 |
| MA-TRM-Lite | 6,965,598 |
| Added parameters | 136,540 |
| Relative increase | 1.9994 percent |

The automated test asserts the exact MA-TRM-Lite count and a permitted range of 6.5 million to 7.5 million parameters.

## Automated tests

The final CPU test suite contains 18 tests. It validates:

- full parameter budget;
- forward tensor shapes;
- private agent-state persistence;
- composite-loss backward propagation;
- router and RecursiveLink gradients;
- mandatory-verifier top-k routing;
- deterministic routing in evaluation;
- sequential, mixture, deliberation, and distillation plans;
- physical top-k execution count;
- batch-level execution-mask consistency;
- zero disagreement for identical predictions;
- answer-change and calibrated Hamming-error helpers;
- synchronous batch-level adaptive halting;
- sample-level halting without restarting completed samples;
- deterministic CPU output for identical seeds;
- inner, link, and joint parameter-freezing policies.

Result:

```text
18 passed
```

## Smoke tests

A forward and backward smoke test completed for all four topologies. Every trainable tensor in the reduced test model received a gradient. The smoke tests report every composite loss component and collaboration metric.

## Synthetic systems benchmark

The benchmark runner completed on CPU with reduced dimensions. This validates timing, synchronization, environment capture, JSON serialization, parameter reporting, output checksums, and peak-memory fields. Synthetic latency is not a task result and should not be cited as evidence of model quality.

## Additional checks

- Python bytecode compilation completed for the new model, training, script, and test modules.
- The paper diagrams compiled from standalone TikZ and were converted to PNG for GitHub rendering.
- The MA-TRM forward path was tested with trainable puzzle identity buffers enabled.
- Dataset hashing, cross-seed metric aggregation, supported-operator FLOP profiling, and shell-script syntax checks completed successfully.
- No em dash characters occur in source, documentation, configuration, or LaTeX text files.
- The original TRM model, dataset builders, and evaluators remain present for direct comparison.

## Validation boundaries

The current execution environment had CPU-only PyTorch. Full CUDA training, distributed NCCL execution, CUDA graph compilation, H100 throughput, and ARC or Sudoku accuracy require the target GPU environment. The repository includes a manual CUDA GitHub Actions workflow and reproducibility scripts for those checks.
