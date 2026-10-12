# Phase 13: Generative AI & LLM Systems Engineering

Welcome to the **LLM Systems Engineering** phase of the Machine Learning Engineer curriculum. This module covers the computational and memory systems underlying modern generative AI: KV-cache memory dynamics, post-training quantization, preference alignment, and hybrid RAG.

---

## 📚 Curriculum & Reference Guides

- [`01_llm_systems_catalog.md`](file:///home/ahmad/ML-Engineer-Training-Materials/13_llm_systems/01_llm_systems_catalog.md): Comprehensive theoretical catalog covering Prefill vs Decoding computational bottlenecks, KV-Cache memory scaling, Uniform INT8 Quantization (Symmetric & Asymmetric), NormalFloat4 (NF4), Direct Preference Optimization (DPO), and Reciprocal Rank Fusion (RRF) Hybrid Search.

---

## 🛠️ Code Implementations (`src/`)

- [`src/kv_cache.py`](file:///home/ahmad/ML-Engineer-Training-Materials/13_llm_systems/src/kv_cache.py): `KVCache` state buffer and `CachedMultiHeadAttention` supporting prefill and token-by-token cached decoding in pure NumPy.
- [`src/quantization.py`](file:///home/ahmad/ML-Engineer-Training-Materials/13_llm_systems/src/quantization.py): `Int8Quantizer` (symmetric/asymmetric calibrated quantization with SNR analysis) and `NormalFloat4Quantizer` (NF4 16-bin optimal Gaussian quantile mapping).
- [`src/dpo_loss.py`](file:///home/ahmad/ML-Engineer-Training-Materials/13_llm_systems/src/dpo_loss.py): `DPOLoss` engine calculating policy preference losses, implicit reward margins, and preference accuracy.
- [`src/hybrid_retrieval.py`](file:///home/ahmad/ML-Engineer-Training-Materials/13_llm_systems/src/hybrid_retrieval.py): `reciprocal_rank_fusion` and `HybridSearchEngine` combining sparse lexical TF-IDF with dense vector embeddings.

---

## 🧪 Unit Tests (`tests/`)

Run the test suite using pytest:

```bash
pytest 13_llm_systems/tests
```
