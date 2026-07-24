# Contributing

## Development setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install einops pydantic hydra-core omegaconf pytest ruff
pytest -q
```

## Pull request requirements

- Add or update tests for every model, routing, loss, or benchmark change.
- Keep the default MA-TRM-Lite core within 6.5 million to 7.5 million trainable parameters.
- Report any change to parameter count, FLOPs, sparse execution, or benchmark protocol.
- Preserve the original TRM baseline so comparisons remain reproducible.
- Do not add claimed benchmark results without configs, seeds, environment manifests, and raw metrics.
- Run `pytest -q` and `ruff check models train scripts tests` before submission.

## Architecture changes

New collaboration topologies should implement `CollaborationTopology.plan`. New latent links should inherit `RecursiveLink`. New loss terms must be logged separately and must have a zero-weight ablation.
