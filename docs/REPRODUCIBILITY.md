# Reproducibility Environment

## Deterministic mode

Set:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DISABLE_COMPILE=1
```

In custom launchers, also call:

```python
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

Deterministic algorithms may reduce throughput. Use one deterministic configuration for scientific comparisons and a separate maximum-throughput configuration for systems profiling.

## Environment capture

```bash
python scripts/collect_environment.py --output benchmarks/environment.json
```

The manifest records the Git commit, dirty status, Python, PyTorch, CUDA runtime, cuDNN, GPU properties, driver output, CPU count, and determinism variables.

## Dataset hashes

After dataset generation, create hashes with:

```bash
find data/arc1concept-aug-1000 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > benchmarks/arc1_dataset_sha256.txt
```

## Run naming

Use a stable run identifier:

```text
{dataset}_{model}_{topology}_params{millions}_seed{seed}_{git-short-sha}
```

Example:

```text
sudoku_ma-trm-lite_sequential_params6.97_seed0_a1b2c3d
```
