# Phase 12: Model Monitoring & Drift Detection

Welcome to the **Model Monitoring & Drift Detection** phase of the Machine Learning Engineer curriculum. This module covers automating statistical vigilance to detect covariate data shift and concept drift in production pipelines before models fail.

---

## 📚 Curriculum & Reference Guides

- [`01_drift_monitoring_catalog.md`](./01_drift_monitoring_catalog.md): Comprehensive theoretical catalog covering Covariate Shift vs Concept Drift, Population Stability Index (PSI), Two-Sample Kolmogorov-Smirnov (K-S) Tests, Wasserstein Distance ($W_1$), Adaptive Windowing (ADWIN), and the Page-Hinkley cumulative sum test.

---

## 🛠️ Code Implementations (`src/`)

- [`src/data_drift.py`](./src/data_drift.py): Vectorized implementations of `population_stability_index`, `ks_two_sample_test`, `wasserstein_distance_1d`, and `DataDriftDetector` multi-feature auditing engine.
- [`src/concept_drift.py`](./src/concept_drift.py): Streaming change detectors featuring `ADWIN` (Adaptive Windowing with Hoeffding split bounds) and `PageHinkley` cumulative sum change detection.

---

## 🧪 Unit Tests (`tests/`)

Run the test suite using pytest:

```bash
pytest 12_model_monitoring_drift/tests
```
