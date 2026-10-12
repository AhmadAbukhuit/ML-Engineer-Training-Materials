# Phase 5: Self-Supervised Learning

Welcome to the **Self-Supervised Learning** phase of the Machine Learning Engineering curriculum. This phase covers representation learning without human annotations: contrastive learning, non-contrastive bootstrap dynamics, masked autoencoders, and pretext tasks.

---

## 📚 Curriculum & Reference Guides

- [`01_self_supervised_models_catalog.md`](./01_self_supervised_models_catalog.md): Comprehensive theoretical catalog of all self-supervised learning models (SimCLR, MoCo, CPC, BYOL, SimSiam, SwAV, Barlow Twins, VICReg, Masked Autoencoders MAE, BEiT, BERT/RoBERTa MLM, GPT CLM, ELECTRA, Denoising Autoencoders, and VAEs).

---

## 🛠️ Code Implementations (`src/`)

- [`src/contrastive_loss.py`](./src/contrastive_loss.py): Normalized Temperature-scaled Cross Entropy (NT-Xent / InfoNCE) loss from first principles with temperature scaling $\tau$, numerical stability, and representation alignment/uniformity metrics.
- [`src/autoencoder.py`](./src/autoencoder.py): Deep Bottleneck Autoencoder with analytical backpropagation, reconstruction MSE loss, gradient clipping, and Denoising Autoencoder (DAE) Gaussian corruption.
- [`src/simclr_framework.py`](./src/simclr_framework.py): SimCLR self-supervised pipeline integrating data augmentation, base encoder $f(x)$, non-linear projection head $g(h)$, and downstream feature extraction.

---

## 🧪 Unit Tests (`tests/`)

Run the self-supervised test suite using pytest:

```bash
pytest 05_self_supervised_learning/tests
```
