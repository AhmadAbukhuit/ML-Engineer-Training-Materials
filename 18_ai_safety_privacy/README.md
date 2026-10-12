# Module 18: AI Safety, Privacy & Algorithmic Fairness

This module provides tools for testing model vulnerability to adversarial perturbations, protecting sensitive data with Differential Privacy, and mathematically auditing demographic disparities in machine learning predictions.

---

## Directory Structure

```text
18_ai_safety_privacy/
├── 01_safety_privacy_catalog.md  # Theoretical definitions & mathematical proofs
├── README.md                     # Usage and verification guide
├── src/
│   ├── __init__.py               # Public exports
│   ├── adversarial.py            # Fast Gradient Sign Method (FGSM) & PGD attacks
│   ├── differential_privacy.py   # DP-SGD gradient clipping & Gaussian mechanism
│   └── fairness_audit.py         # Demographic Parity, Equalized Odds & Disparate Impact
└── tests/
    ├── __init__.py
    ├── test_adversarial.py       # Epsilon perturbation & projection bounds tests
    ├── test_differential_privacy.py # L2 gradient clipping and noise tests
    └── test_fairness_audit.py    # Metric parity and EEOC 80% rule tests
```

---

## Quickstart Example

```python
import numpy as np
from src.adversarial import FGSM
from src.differential_privacy import DPSGDStep
from src.fairness_audit import FairnessAuditor

# 1. FGSM Adversarial Perturbation
fgsm = FGSM(epsilon=0.1, clip_min=0.0, clip_max=1.0)
x_clean = np.array([0.5, 0.2])
grad_fn = lambda x: np.array([1.0, -1.0])
x_adv = fgsm.generate(x_clean, grad_fn)
print("Adversarial Sample:", x_adv)

# 2. Differentially Private SGD Step
dp = DPSGDStep(clipping_norm=1.0, noise_multiplier=0.5)
per_sample_grads = np.random.randn(32, 10)
priv_grad, clip_ratio = dp.privatize_batch_gradients(per_sample_grads)
print("Privatized Gradient shape:", priv_grad.shape, "Clipping Ratio:", clip_ratio)

# 3. Algorithmic Fairness Audit
y_true = np.array([1, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 1, 0])
sensitive_group = np.array([0, 0, 0, 1, 1, 1])
report = FairnessAuditor.audit(y_true, y_pred, sensitive_group)
print("Demographic Parity Difference:", report.demographic_parity_diff)
print("Disparate Impact Ratio:", report.disparate_impact_ratio)
```

---

## Running Verification Tests

```bash
pytest 18_ai_safety_privacy/tests
```
