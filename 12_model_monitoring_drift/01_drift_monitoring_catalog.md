# Complete Catalog of Model Monitoring & Drift Detection

Once machine learning models are deployed to production, their predictive performance inevitably degrades over time due to non-stationary environments, evolving user behaviors, upstream data pipeline corruptions, and economic macro-shifts. Model monitoring provides automated statistical vigilance to detect covariate data drift and concept drift before silent business failures occur.

---

## 1. Taxonomy of Dataset Shift

Under the probabilistic framework of joint distribution $P(X, Y) = P(X) P(Y|X) = P(Y) P(X|Y)$:

| Shift Type | Formal Mathematical Definition | Real-World Scenario | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Covariate Shift (Data Drift)** | $P_{\text{train}}(X) \ne P_{\text{prod}}(X)$, while $P(Y\|X)$ remains unchanged | New demographics or new smartphone models enter the platform; input features drift but physical relationships hold. | Importance weighting, domain adaptation, feature realignment. |
| **Concept Drift** | $P_{\text{train}}(Y\|X) \ne P_{\text{prod}}(Y\|X)$, while $P(X)$ may or may not change | A global pandemic or financial crisis changes consumer purchasing rules; identical inputs produce different outcomes. | Model retraining on recent window, online continual learning. |
| **Prior Probability Shift** | $P_{\text{train}}(Y) \ne P_{\text{prod}}(Y)$, while $P(X\|Y)$ remains unchanged | Seasonal flu epidemic surges the fraction of positive diagnoses, while symptoms per infection stay the same. | Prior reweighting, decision threshold calibration. |

---

## 2. Statistical Tests for Covariate Data Drift

### 2.1 Population Stability Index (PSI)

- **Origin**: Credit scoring and enterprise risk management.
- **Mathematical Formulation**:
  Binned into $K$ quantile buckets:
  $$\text{PSI} = \sum_{k=1}^K \left( Q_k - P_k \right) \ln\left( \frac{Q_k}{P_k} \right)$$
  where $P_k$ is the reference/training frequency fraction in bucket $k$, and $Q_k$ is the actual production frequency fraction.
- **Symmetric Kullback-Leibler Form**:
  $$\text{PSI} = D_{\text{KL}}(Q \parallel P) + D_{\text{KL}}(P \parallel Q)$$
- **Standard Industrial Thresholds**:
  - $\text{PSI} < 0.1$: Insignificant shift; baseline stable.
  - $0.1 \le \text{PSI} < 0.2$: Moderate shift; trigger automated warning and diagnostic report.
  - $\text{PSI} \ge 0.2$: Severe distribution drift; automated model retraining or fallback to heuristic baseline required.

### 2.2 Two-Sample Kolmogorov-Smirnov (K-S) Test

- **Non-Parametric**: Requires zero parametric assumptions about underlying distribution families.
- **Test Statistic**:
  $$D_{n_1, n_2} = \sup_x |F_1(x) - F_2(x)|$$
  Measures the maximum vertical separation between the empirical cumulative distribution functions (ECDF) of two continuous samples.
- **Decision Rule**: Null hypothesis $H_0: F_1 = F_2$. If asymptotic $p$-value $< \alpha$ (e.g. 0.05), reject $H_0$ and declare statistically significant drift.

### 2.3 Wasserstein Distance (Earth Mover's Distance - $W_1$)

- Evaluates the minimum work needed to transform one probability distribution into another:
  $$W_1(u, v) = \int_{-\infty}^\infty |U(x) - V(x)| \, dx$$
- In contrast to KL divergence, Wasserstein distance remains well-defined, continuous, and non-infinite even when distributions have non-overlapping supports.

---

## 3. Streaming Concept Drift Detection

Unlike data drift (which only requires unlabelled feature inputs $X$), concept drift monitoring tracks changes in model accuracy or residuals $(y, \hat{y})$ in real-time streaming environments.

### 3.1 Adaptive Windowing (ADWIN - Bifet & Gavaldà, 2007)

- **Algorithm Mechanics**:
  1. Maintains an active window $W$ of recent sample observations.
  2. Dynamically evaluates all possible binary split partitions $W = W_0 \cdot W_1$.
  3. If the difference in sub-window means exceeds a Hoeffding cut bound:
     $$|\hat{\mu}_{W_0} - \hat{\mu}_{W_1}| \ge \epsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \ln\left(\frac{4}{\delta'}\right)}$$
     where $m = (\frac{1}{|W_0|} + \frac{1}{|W_1|})^{-1}$ is the harmonic sample size and $\delta' = \delta / |W|$.
  4. ADWIN flags drift and drops the older sub-window $W_0$.
- **Guarantees**: Rigorous bounds on both false positive rate (at most $\delta$) and detection delay.

### 3.2 Page-Hinkley Test

- Sequential cumulative sum test designed to detect abrupt jumps in streaming error metrics.
- Updates running mean $\bar{x}_t$ and cumulative sum:
  $$U_t = \sum_{k=1}^t (x_k - \bar{x}_k - \alpha), \quad m_t = \min_{1 \le k \le t} U_k$$
- Triggers alert when difference $U_t - m_t > \lambda$.

---

## 4. Production Architectural Implementation

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        MLOBSERVABILITY PIPELINE                        │
├─────────────────────┬───────────────────────────┬──────────────────────┤
│ Offline Training    │ Online Serving            │ Automated Action     │
│ ─────────────────── │ ───────────────────────── │ ──────────────────── │
│ • Compute Reference │ • Buffer Production Batch │ • PSI < 0.1: Normal  │
│   Quantiles & ECDFs │ • Compute PSI & KS-Stats  │ • 0.1 <= PSI < 0.2:  │
│ • Store Baseline    │ • Monitor Residual Error  │   Warning Alert      │
│   Statistics        │   via ADWIN               │ • PSI >= 0.2: Trigger│
│                     │                           │   Retraining Pipeline│
└─────────────────────┴───────────────────────────┴──────────────────────┘
```
