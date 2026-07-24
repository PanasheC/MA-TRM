# MA-TRM-Lite Model Card

## Model summary

MA-TRM-Lite is a non-autoregressive recursive neural model for exact structured reasoning. It extends the Tiny Recursive Model with four role-conditioned reasoners that share one dense backbone. The roles maintain private latent states and communicate through trainable low-rank residual links.

## Intended uses

- Sudoku, maze, ARC-AGI, and similar finite structured reasoning benchmarks.
- Research on recursive depth, latent collaboration, adaptive computation, and parameter sharing.
- Parameter-matched and compute-matched comparisons with the included TRM baseline.
- Ablation studies of routing, private state, communication, topology, verification, and cell attention.

## Out-of-scope uses

- Safety-critical decision support.
- Open-ended language generation.
- Claims of general intelligence from benchmark accuracy alone.
- Production deployment without task-specific validation and monitoring.

## Default architecture

| Property | Value |
|---|---|
| Trainable parameters | 6,965,598 for vocabulary size 12 |
| Shared backbone | 2 layers, width 512 |
| Roles | pattern, transform, critic, verifier |
| Adapter rank | 8 |
| RecursiveLink rank | 8 |
| Active roles at inference | top 2, verifier mandatory |
| Collaboration topology | sequential |
| Collaboration rounds per ACT call | 3 |
| Local cycles per active role | 2 |
| Maximum ACT calls | 16 |
| Precision | bfloat16 on supported GPUs |

## Training data

The code does not ship trained weights. Users generate datasets with the inherited TRM builders. Dataset generation settings, augmentation, split selection, task identity buffers, and hashes must accompany every reported result.

## Evaluation requirements

Report exact accuracy, cell accuracy, calibration, trainable parameters, puzzle identity buffer size, measured FLOPs, wall time, peak VRAM, mean ACT calls, active roles, active cells, seeds, software versions, and hardware.

## Risks and limitations

- Specialized roles can collapse into similar representations.
- Low disagreement can occur when all roles agree on the same error.
- Batch-level physical routing can hide per-instance routing diversity.
- The verifier's expected Hamming error interpretation requires calibration on the evaluation distribution.
- Adaptive recursion can trade accuracy for latency if thresholds are tuned on the test set.
- Puzzle identity embeddings can inflate effective capacity and weaken claims about unseen-task generalization.
- Exact reproducibility across GPU architectures and PyTorch versions is not guaranteed, even with fixed seeds.

## Licensing and attribution

MIT. The repository retains the original TinyRecursiveModels copyright and license notice. See `NOTICE` and `LICENSE`.
