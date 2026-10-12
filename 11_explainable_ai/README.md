# Phase 11: Explainable AI (XAI) & Model Interpretability

Welcome to the **Explainable AI (XAI)** phase of the Machine Learning Engineer curriculum. This module covers transforming opaque predictive models into transparent, auditable, and mathematically grounded decision systems.

---

## 📚 Curriculum & Reference Guides

- [`01_explainable_ai_catalog.md`](./01_explainable_ai_catalog.md): Comprehensive theoretical guide covering Cooperative Game Theory, Shapley Axioms (Efficiency, Symmetry, Dummy, Additivity), KernelSHAP, TreeSHAP, LIME local linear surrogates, Permutation Feature Importance, and Partial Dependence / ICE curves.

---

## 🛠️ Code Implementations (`src/`)

- [`src/shapley_values.py`](./src/shapley_values.py): `ExactShapleyExplainer` via coalition powerset enumeration and `SamplingShapleyExplainer` via Monte Carlo random permutations.
- [`src/lime_explainer.py`](./src/lime_explainer.py): `LimeTabularExplainer` generating Gaussian neighborhood perturbations, exponential distance weighting $\pi_x(z)$, and weighted Ridge regression local surrogates.
- [`src/permutation_importance.py`](./src/permutation_importance.py): Vectorized permutation feature importance quantifying out-of-sample metric drops across repetitions.
- [`src/partial_dependence.py`](./src/partial_dependence.py): Partial Dependence (PDP) marginal expectations and Individual Conditional Expectation (ICE) curves.

---

## 🧪 Unit Tests (`tests/`)

Run the test suite using pytest:

```bash
pytest 11_explainable_ai/tests
```
