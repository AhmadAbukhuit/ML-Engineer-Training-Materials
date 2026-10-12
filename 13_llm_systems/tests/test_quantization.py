import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantization import Int8Quantizer, NormalFloat4Quantizer


def test_int8_symmetric_quantization():
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 0.5, size=(64, 64))

    q_engine = Int8Quantizer(mode="symmetric").fit_calibrate(weights)
    q_weights = q_engine.quantize(weights)

    assert q_weights.dtype == np.int8
    assert np.min(q_weights) >= -128
    assert np.max(q_weights) <= 127

    err = q_engine.quantization_error(weights)
    assert err["rmse"] < 0.01
    assert err["snr_db"] > 35.0  # High fidelity SNR


def test_int8_asymmetric_quantization():
    activations = np.random.uniform(0.0, 10.0, size=(100, 32))

    q_engine = Int8Quantizer(mode="asymmetric").fit_calibrate(activations)
    q_act = q_engine.quantize(activations)

    assert q_act.dtype == np.uint8
    assert np.min(q_act) >= 0
    assert np.max(q_act) <= 255

    err = q_engine.quantization_error(activations)
    assert err["rmse"] < 0.05


def test_nf4_quantization():
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 1, size=(128, 128))

    nf4 = NormalFloat4Quantizer()
    q_indices, absmax = nf4.quantize(weights)

    assert q_indices.dtype == np.uint8
    assert np.max(q_indices) <= 15
    assert np.min(q_indices) >= 0

    recon = nf4.dequantize(q_indices, absmax)
    rmse = float(np.sqrt(np.mean((weights - recon) ** 2)))
    assert rmse < 0.20  # 4-bit lossy quantization bound
