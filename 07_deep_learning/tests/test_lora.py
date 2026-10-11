import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lora import LoRALinear


def test_lora_initialization_and_forward():
    in_features = 32
    out_features = 16
    rank = 4
    alpha = 8.0

    lora_layer = LoRALinear(in_features, out_features, rank=rank, alpha=alpha, random_state=42)

    # B initialized to zeros -> initial output matches frozen base output
    x = np.random.randn(8, in_features)
    out = lora_layer.forward(x)
    base_out = np.matmul(x, lora_layer.W_0) + lora_layer.bias

    assert out.shape == (8, out_features)
    assert np.allclose(out, base_out)

    # Check parameter reduction
    param_info = lora_layer.parameter_counts()
    assert param_info["trainable_lora_params"] == (32 * 4) + (4 * 16)  # 192
    assert param_info["base_params"] == (32 * 16) + 16  # 528
    assert param_info["reduction_ratio"] > 1.0


def test_lora_gradient_and_update():
    lora_layer = LoRALinear(in_features=8, out_features=4, rank=2, alpha=4.0, random_state=42)
    x = np.random.randn(4, 8)
    _ = lora_layer.forward(x)

    initial_W_0 = lora_layer.W_0.copy()
    initial_B = lora_layer.lora_B.copy()

    # Backprop dummy grad
    grad_out = np.ones((4, 4))
    grad_in = lora_layer.backward(grad_out)

    assert grad_in.shape == (4, 8)
    assert lora_layer.grad_A is not None
    assert lora_layer.grad_B is not None

    # Update LoRA parameters
    lora_layer.update_lora(lr=0.1)

    # Base weights must remain untouched
    assert np.allclose(lora_layer.W_0, initial_W_0)
    # LoRA parameters must be modified
    assert not np.allclose(lora_layer.lora_B, initial_B)


def test_lora_weight_merging():
    lora_layer = LoRALinear(in_features=8, out_features=4, rank=2, alpha=4.0, random_state=42)
    # Put non-zero values into B to test difference
    lora_layer.lora_B = np.random.randn(2, 4)

    x = np.random.randn(5, 8)
    unmerged_out = lora_layer.forward(x)

    # Merge weights
    lora_layer.merge_weights()
    assert lora_layer.merged
    merged_out = lora_layer.forward(x)

    # Inference output must be identical before and after merging
    assert np.allclose(unmerged_out, merged_out, atol=1e-7)

    # Unmerge and verify
    lora_layer.unmerge_weights()
    assert not lora_layer.merged
    unmerged_again_out = lora_layer.forward(x)
    assert np.allclose(unmerged_out, unmerged_again_out, atol=1e-7)
