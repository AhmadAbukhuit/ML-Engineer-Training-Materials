"""Tests for ZeRO Memory Estimator."""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from zero_memory import ZeROMemoryEstimator


def test_zero_memory_profiles():
    # 1 Billion parameter model on 8 GPUs
    n_params = 1_000_000_000
    world_size = 8
    estimator = ZeROMemoryEstimator(n_params, world_size)

    ddp = estimator.baseline_ddp()
    z1 = estimator.zero_stage_1()
    z2 = estimator.zero_stage_2()
    z3 = estimator.zero_stage_3()

    # DDP static footprint = 16 bytes * 10^9 = 16 GB (approx 14.90 GiB)
    assert ddp.total_static_gb > 14.0

    # Strict hierarchy of memory footprint reduction
    assert ddp.total_static_gb > z1.total_static_gb
    assert z1.total_static_gb > z2.total_static_gb
    assert z2.total_static_gb > z3.total_static_gb

    # ZeRO-3 should achieve exactly 8x reduction over DDP
    assert z3.total_static_gb == pytest.approx(ddp.total_static_gb / world_size, rel=1e-5)
