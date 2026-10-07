# Support Vector Machines, Naive Bayes & Time Series Forecasting

This guide details the mathematical principles behind Support Vector Machines (SVMs), Gaussian Naive Bayes, Modern Gradient Boosted Decision Trees (GBDTs), and Time Series Forecasting.

---

## 1. Support Vector Machines (SVM)

Support Vector Machines find the hyperplane that maximizes the geometric **margin** (the distance to the nearest training samples, called *support vectors*).

### The Optimization Formulation

For linearly separable data labeled $y_i \in \{-1, +1\}$, the hard-margin objective is:

$$\min_{\mathbf{w}, b} \frac{1}{2} \Vert{}\mathbf{w}\Vert{}_2^2 \quad \text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad \forall i$$

### Soft-Margin SVM & Hinge Loss

In real-world data, classes overlap. We introduce slack variables $\xi_i \ge 0$ to allow margin violations:

$$\mathcal{L}(\mathbf{w}, b) = \frac{\lambda}{2} \Vert{}\mathbf{w}\Vert{}_2^2 + \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b))$$

* If a point is correctly classified outside the margin ($y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1$), its hinge loss is $0$.
* If a point violates the margin, the hinge loss increases linearly with the distance from the margin boundary.

### The Kernel Trick

To separate non-linear data, we map samples $\mathbf{x} \mapsto \phi(\mathbf{x})$ into a higher-dimensional Hilbert space where they become linearly separable.
By **Mercer's Theorem**, we do not need to compute $\phi(\mathbf{x})$ explicitly; we only need the inner-product kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$:

* **Linear Kernel:** $K(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T \mathbf{v}$
* **Polynomial Kernel:** $K(\mathbf{u}, \mathbf{v}) = (\mathbf{u}^T \mathbf{v} + c)^d$
* **Radial Basis Function (RBF) Kernel:** $K(\mathbf{u}, \mathbf{v}) = \exp\left(-\gamma \Vert{}\mathbf{u} - \mathbf{v}\Vert{}^2\right)$ *(corresponds to an infinite-dimensional feature space)*

---

## 2. Gaussian Naive Bayes

Naive Bayes is a generative probabilistic classifier applying Bayes' Theorem with the strong **conditional independence assumption**: all features are assumed to be mutually independent given the class label.

$$P(y=c \mid \mathbf{x}) = \frac{P(y=c) \prod_{j=1}^D P(x_j \mid y=c)}{P(\mathbf{x})}$$

For continuous variables, each feature likelihood is modeled as a normal distribution:

$$P(x_j \mid y=c) = \frac{1}{\sqrt{2\pi \sigma_{c,j}^2}} \exp\left( -\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right)$$

* **Log-Space Computation:** Because multiplying hundreds of probabilities between $0$ and $1$ causes numerical underflow, we sum log-probabilities:
  $$\log P(y=c \mid \mathbf{x}) \propto \log P(y=c) + \sum_{j=1}^D \log P(x_j \mid y=c)$$

---

## 3. Time Series Forecasting: Stationarity & Walk-Forward Validation

Time series data exhibits temporal dependency. Standard machine learning cross-validation (random shuffling) creates **lookahead bias** (leaking future information into past predictions).

### Feature Engineering for Temporal Sequences

1. **Lag Features:** $y_{t-1}, y_{t-2}, \dots, y_{t-k}$ allow linear and tree models to capture autoregressive behavior.
2. **Rolling Statistics:** $\text{mean}(y_{t-1 \dots t-w})$ and $\text{std}(y_{t-1 \dots t-w})$ capture local trend and volatility.
3. **Calendar/Cyclical Encodings:** Hours of the day or months are periodic. Transforming them into sine/cosine pairs prevents the false assumption that hour 23 and hour 0 are far apart:
   $$x_{\sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$

### Walk-Forward (Rolling-Origin) Cross-Validation

```text
Iteration 1: [ Train: t=1..30 ] -> [ Test: t=31..40 ]
Iteration 2: [ Train: t=1..40 ] -> [ Test: t=41..50 ]
Iteration 3: [ Train: t=1..50 ] -> [ Test: t=51..60 ]
```

Every test split is strictly in the future relative to its corresponding training split.
