# Phase 9: Transfer Learning & Fine-Tuning

Welcome to the **Transfer Learning** phase of the Machine Learning Engineering curriculum. This phase covers leveraging pretrained representations for downstream tasks: feature extraction backbones, gradual unfreezing, Parameter-Efficient Fine-Tuning (LoRA, QLoRA, Adapters), weight merging for zero inference latency, and Domain Adaptation via Maximum Mean Discrepancy.

---

## 📚 Curriculum & Reference Guides

- [`01_transfer_learning_models_catalog.md`](./01_transfer_learning_models_catalog.md): Comprehensive theoretical catalog of all transfer learning algorithms (Linear Probing, Discriminative Fine-Tuning, LLRD, Catastrophic Forgetting mitigation, LoRA, QLoRA, Adapters, Prompt/Prefix Tuning, $(IA)^3$, MMD Domain Adaptation, and DANN).

---

## 🛠️ Code Implementations (`src/`)

- [`src/transfer_learning.py`](./src/transfer_learning.py): Two-stage transfer learning engine featuring backbone parameter freezing, classification head replacement, and fine-tuning.
- [`src/lora.py`](./src/lora.py): Low-Rank Adaptation (LoRA) layer decomposing updates into low-rank matrices $A$ and $B$, parameter accounting, and weight merging ($W_{\text{merged}} = W_0 + \frac{\alpha}{r}AB$) for zero-overhead inference.
- [`src/domain_adaptation.py`](./src/domain_adaptation.py): Maximum Mean Discrepancy (MMD) distribution alignment with multi-kernel Gaussian RBF.

---

## 🧪 Unit Tests (`tests/`)

Run the transfer learning test suite using pytest:

```bash
pytest 09_transfer_learning/tests
```
