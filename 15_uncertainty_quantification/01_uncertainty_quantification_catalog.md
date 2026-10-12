# Complete Catalog of Uncertainty Quantification & Conformal Prediction

Machine learning systems operating in safety-critical domains (autonomous driving, medical diagnostics, quantitative finance) cannot rely solely on point predictions. They require calibrated, mathematically grounded measures of uncertainty.

---

## 1. Distribution-Free Uncertainty: Conformal Prediction

Unlike Bayesian methods that assume specific prior or noise distributions, **Conformal Prediction (Vovk et al., 2005)** provides rigorous, distribution-free, finite-sample predictive guarantees under the sole assumption that data points are independent and identically distributed (or exchangeable).

### 1.1 Inductive / Split Conformal Prediction

Given training data $\mathcal{D}_{\text{train}}$, a held-out calibration set $\mathcal{D}_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^n$, and a target significance level $\alpha \in (0, 1)$ (desired coverage $1 - \alpha$):

1. Train an arbitrary base predictor $\hat{f}$ on $\mathcal{D}_{\text{train}}$.
2. For each calibration sample $(x_i, y_i) \in \mathcal{D}_{\text{cal}}$, compute non-conformity score:
   $$s_i = |y_i - \hat{f}(x_i)|$$
3. Compute the finite-sample adjusted quantile:
   $$\hat{q} = \text{Quantile}\left( s_{1:n}, \, \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n} \right)$$
4. For any new test input $x_{n+1}$, form prediction interval:
   $$C(x_{n+1}) = \left[ \hat{f}(x_{n+1}) - \hat{q}, \, \hat{f}(x_{n+1}) + \hat{q} \right]$$

### 1.2 Mathematical Guarantee

Under data exchangeability, the marginal coverage satisfies:

$$1 - \alpha \le P\left( Y_{n+1} \in C(X_{n+1}) \right) \le 1 - \alpha + \frac{1}{n + 1}$$

This guarantee holds for *any* sample size $n$, *any* base algorithm (Deep Neural Nets, XGBoost, Random Forests), and *any* underlying data distribution.

---

## 2. Epistemic vs. Aleatoric Uncertainty: Bayesian Linear Regression

Predictive uncertainty decomposes into two distinct components:

- **Aleatoric Uncertainty (Data Noise):** Inherent, irreducible randomness in the physical process generating the data ($\sigma^2_{\text{noise}}$).
- **Epistemic Uncertainty (Model / Parameter Uncertainty):** Lack of knowledge due to sparse observations in regions of the feature space; reducible by collecting more data.

### 2.1 Conjugate Gaussian Prior & Posterior

For linear model $y = w^T x + \epsilon$ with known noise precision $\beta$ ($\epsilon \sim \mathcal{N}(0, \beta^{-1})$) and isotropic prior $p(w) = \mathcal{N}(0, \alpha^{-1} I)$:

$$\text{Posterior Precision:} \quad S_N^{-1} = \alpha I + \beta \Phi^T \Phi$$
$$\text{Posterior Mean:} \quad m_N = \beta S_N \Phi^T y$$

### 2.2 Posterior Predictive Distribution

For an arbitrary query $x_*$, the predictive distribution is Gaussian:

$$p(y_* | x_*, X, y) = \mathcal{N}\left( m_N^T x_*, \, \sigma^2_N(x_*) \right)$$
$$\sigma^2_N(x_*) = \underbrace{\frac{1}{\beta}}_{\text{Aleatoric Variance}} + \underbrace{x_*^T S_N x_*}_{\text{Epistemic Variance}}$$

When $x_*$ lies far outside the support of training data $\Phi$, $x_*^T S_N x_* \to \infty$, correctly alerting downstream systems of out-of-distribution risks.

---

## 3. Non-Parametric Bayesian Modeling: Gaussian Process Regression (GPR)

A Gaussian Process is a collection of infinite random variables, any finite subset of which has a joint multivariate Gaussian distribution: $f(x) \sim \mathcal{GP}(m(x), k(x, x'))$.

### 3.1 The Squared Exponential (RBF) Kernel

Captures smooth, continuous non-linear functions:

$$k(x, x') = \sigma_f^2 \exp\left( -\frac{\|x - x'\|^2}{2\ell^2} \right)$$

where $\ell$ controls horizontal correlation scale and $\sigma_f^2$ controls output amplitude.

### 3.2 Exact Inference via Cholesky Factorization

With noisy observations $y = f(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$:

- Let $K = k(X, X)$ and $K_y = K + \sigma_n^2 I$.
- Decompose $K_y = L L^T$ via Cholesky factorization in $O(n^3)$ time.
- Posterior Mean: $\mu_* = K_*^T K_y^{-1} y = K_*^T (L^T \backslash (L \backslash y))$
- Posterior Variance: $\Sigma_* = K_{**} - K_*^T K_y^{-1} K_* = K_{**} - v^T v$, where $v = L \backslash K_*$.

GP predictions automatically interpolate training points with minimal uncertainty while expanding variance in data-sparse intervals.
