# Module 15: Uncertainty Quantification & Conformal Prediction

This module covers foundational algorithms for estimating predictive confidence, decomposing epistemic versus aleatoric risk, and constructing distribution-free prediction intervals with exact finite-sample coverage guarantees.

---

## Directory Structure

```text
15_uncertainty_quantification/
├── 01_uncertainty_quantification_catalog.md  # Theoretical foundations and mathematical proofs
├── README.md                                 # Architecture and usage guide
├── src/
│   ├── __init__.py                           # Public API exports
│   ├── conformal_prediction.py               # Split Conformal regression & classification sets
│   ├── bayesian_linear_regression.py         # Conjugate prior/posterior & variance decomposition
│   └── gaussian_process.py                   # RBF kernel GP with Cholesky solve
└── tests/
    ├── __init__.py
    ├── test_conformal_prediction.py          # Marginal coverage guarantees tests
    ├── test_bayesian_linear_regression.py    # OOD uncertainty expansion tests
    └── test_gaussian_process.py              # Exact interpolation and gap variance tests
```

---

## Quickstart Example

```python
import numpy as np
from src.conformal_prediction import SplitConformalRegressor
from src.gaussian_process import GaussianProcessRegressor

# 1. Distribution-free Conformal Prediction
class SimpleModel:
    def predict(self, X):
        return 2.0 * X[:, 0]

model = SimpleModel()
X_cal = np.random.randn(200, 1)
y_cal = 2.0 * X_cal[:, 0] + np.random.normal(scale=0.3, size=200)

conformal = SplitConformalRegressor(model).calibrate(X_cal, y_cal, alpha=0.1)
lower, upper = conformal.predict_interval(np.array([[1.5], [-2.0]]))
print("Prediction intervals [lower, upper]:", list(zip(lower, upper)))

# 2. Gaussian Process Epistemic Uncertainty
X_train = np.array([[-1.0], [0.0], [1.0]])
y_train = np.sin(X_train[:, 0])

gp = GaussianProcessRegressor(length_scale=1.0, noise_variance=1e-4).fit(X_train, y_train)
mu, std = gp.predict(np.array([[0.0], [5.0]]), return_std=True)
print("GP Mean:", mu)
print("GP Uncertainty (higher at x=5.0):", std)
```

---

## Running Verification Tests

```bash
pytest 15_uncertainty_quantification/tests
```
