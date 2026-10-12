import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kv_cache import CachedMultiHeadAttention, KVCache


def test_kv_cache_accumulation():
    cache = KVCache()
    assert cache.seq_len == 0

    k1 = np.random.randn(2, 4, 3, 16)
    v1 = np.random.randn(2, 4, 3, 16)
    full_k, _ = cache.update(k1, v1)
    assert cache.seq_len == 3
    assert full_k.shape == (2, 4, 3, 16)

    # Append single token
    k2 = np.random.randn(2, 4, 1, 16)
    v2 = np.random.randn(2, 4, 1, 16)
    full_k, _ = cache.update(k2, v2)
    assert cache.seq_len == 4
    assert full_k.shape == (2, 4, 4, 16)


def test_cached_mha_equivalence():
    mha = CachedMultiHeadAttention(d_model=32, num_heads=4, random_state=42)
    prompt = np.random.randn(1, 5, 32)

    # Prefill pass
    cache = KVCache()
    out_prefill, _ = mha.forward(prompt, kv_cache=cache)
    assert out_prefill.shape == (1, 5, 32)
    assert cache.seq_len == 5

    # Decode step: new token (shape: 1, 1, 32)
    new_token = np.random.randn(1, 1, 32)
    out_token, attn_token = mha.forward(new_token, kv_cache=cache)

    assert out_token.shape == (1, 1, 32)
    assert cache.seq_len == 6
    assert attn_token.shape == (1, 4, 1, 6)
