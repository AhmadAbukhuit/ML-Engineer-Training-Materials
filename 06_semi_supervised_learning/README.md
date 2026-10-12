# Phase 6: Semi-Supervised Learning

Welcome to the **Semi-Supervised Learning** phase of the Machine Learning Engineering curriculum. This phase covers algorithms that exploit small labeled datasets in combination with massive unlabeled corpora: pseudo-labeling, consistency regularization, graph affinity propagation, and generative semi-supervised models.

---

## 📚 Curriculum & Reference Guides

- [`01_semi_supervised_models_catalog.md`](./01_semi_supervised_models_catalog.md): Comprehensive theoretical catalog of all semi-supervised algorithms (Self-Training, FixMatch, FlexMatch, $\Pi$-Model, Temporal Ensembling, Mean Teacher, UDA, MixMatch, Label Propagation, Label Spreading, GCNs, Semi-supervised VAE/GAN, and Transductive SVMs).

---

## 🛠️ Code Implementations (`src/`)

- [`src/pseudo_labeling.py`](./src/pseudo_labeling.py): Self-Training / Pseudo-Labeling Classifier wrapper from first principles with confidence thresholding $\tau$ and iterative dataset expansion.
- [`src/label_propagation.py`](./src/label_propagation.py): Graph-Based Label Propagation algorithm from scratch via RBF kernel affinity matrices, Markov transition normalization, and ground-truth node clamping.
- [`src/consistency_regularization.py`](./src/consistency_regularization.py): Consistency regularization utilities including probability sharpening ($q_i \propto p_i^{1/T}$), Gaussian ramp-up schedules, and MSE/KL consistency loss computation.

---

## 🧪 Unit Tests (`tests/`)

Run the semi-supervised test suite using pytest:

```bash
pytest 06_semi_supervised_learning/tests
```
