# Module 17: Distributed Deep Learning & Parallelism

This module covers the core architectural and systems engineering principles required for distributed model training across multi-node GPU clusters.

---

## Directory Structure

```
17_distributed_training/
├── 01_distributed_training_catalog.md  # Detailed communication & memory sharding derivations
├── README.md                           # Usage and testing guide
├── src/
│   ├── __init__.py                     # Public API exports
│   ├── ring_allreduce.py               # Step-by-step Ring All-Reduce communication simulation
│   ├── zero_memory.py                  # Exact byte-level ZeRO-1/2/3 memory accounting
│   └── loss_scaler.py                  # Dynamic Loss Scaler for FP16 Mixed Precision
└── tests/
    ├── __init__.py
    ├── test_ring_allreduce.py          # Ring communication correctness across cluster sizes
    ├── test_zero_memory.py             # Memory reduction hierarchy tests
    └── test_loss_scaler.py             # Overflow backoff and growth interval tests
```

---

## Quickstart Example

```python
import numpy as np
from src.ring_allreduce import ring_allreduce
from src.zero_memory import ZeROMemoryEstimator
from src.loss_scaler import DynamicLossScaler

# 1. Ring All-Reduce across 4 simulated ranks
rank_tensors = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]), np.array([7.0, 8.0])]
reduced = ring_allreduce(rank_tensors)
print("Globally reduced array on Rank 0:", reduced[0])  # [16.0, 20.0]

# 2. ZeRO Memory Breakdown for 7B Parameter Model on 8 GPUs
zero = ZeROMemoryEstimator(num_parameters=7_000_000_000, world_size=8)
print("Baseline DDP Per-GPU Memory:", round(zero.baseline_ddp().total_static_gb, 2), "GB")
print("ZeRO-3 Per-GPU Memory:", round(zero.zero_stage_3().total_static_gb, 2), "GB")

# 3. Dynamic Loss Scaler
scaler = DynamicLossScaler(initial_scale=65536.0)
scaled_loss = scaler.scale_loss(0.45)
raw_grads = [np.array([13107.2, 26214.4])]
unscaled_grads, is_finite = scaler.unscale_gradients(raw_grads)
print("Unscaled Gradients:", unscaled_grads)
```

---

## Running Verification Tests

```bash
pytest 17_distributed_training/tests
```
