# Complete Catalog of Supervised Learning Models

Supervised Learning is the machine learning paradigm where an algorithm learns a mapping function $f: \mathcal{X} \to \mathcal{Y}$ from a labeled dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, such that the empirical risk (or regularized risk) over a chosen loss function $\mathcal{L}(\hat{y}, y)$ is minimized:

$$\min_{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(\mathbf{x}_i; \boldsymbol{\theta}), y_i) + \lambda \mathcal{R}(\boldsymbol{\theta})$$

---

## 1. Linear & Generalized Linear Models

### 1.1 Ordinary Least Squares (OLS) Linear Regression
- **Mathematical Formulation**: $\hat{y} = \mathbf{w}^T \mathbf{x} + b$.
- **Objective Function**: Mean Squared Error (MSE):
  $$J(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^N (\mathbf{w}^T \mathbf{x}_i + b - y_i)^2 = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$$
- **Closed-Form Normal Equation**:
  $$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
- **Optimization**: Analytical inversion (for $D \le 10^4$) or Gradient Descent / Mini-batch SGD (for large scale $N, D$).
- **Assumptions**: Linearity, homoscedasticity, normality of residuals, independence of observations, no multicollinearity.

### 1.2 Ridge Regression ($L_2$ Regularization)
- **Objective Function**:
  $$J(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
- **Closed-Form**: $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$.
- **Properties**: Prevents multicollinearity instability; shrinks weights towards zero without making them exactly zero; convex and strictly differentiable everywhere.

### 1.3 Lasso Regression ($L_1$ Regularization)
- **Objective Function**:
  $$J(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_1$$
- **Properties**: Non-smooth at $w_j = 0$; diamond-shaped constraint geometry produces exact sparsity (feature selection); solved via Subgradient Descent, Coordinate Descent, or Proximal Gradient Methods (ISTA/FISTA).

### 1.4 ElasticNet
- **Objective Function**: Linear combination of $L_1$ and $L_2$ penalties:
  $$J(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \frac{\alpha(1-\rho)}{2} \|\mathbf{w}\|_2^2$$
- **Properties**: Overcomes Lasso's limitation when $p > n$ and groups correlated features together rather than arbitrarily picking one.

### 1.5 Logistic Regression (Binary & Multinomial)
- **Binary Formulation**:
  $$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Objective Function**: Binary Cross-Entropy (Negative Log-Likelihood):
  $$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$
- **Multinomial (Softmax Regression)**:
  $$P(y=k|\mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x})}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x})}$$
- **Optimization**: L-BFGS, Newton-Raphson (IRLS - Iteratively Reweighted Least Squares), SGD.

---

## 2. Support Vector Machines (SVM)

### 2.1 Hard-Margin SVM
- Maximizes the margin $\frac{2}{\|\mathbf{w}\|_2}$ subject to perfect linear separation:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \; \forall i$$

### 2.2 Soft-Margin SVM (C-SVM)
- Introduces slack variables $\xi_i \ge 0$ for non-linearly separable data:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \; \xi_i \ge 0$$
- **Unconstrained Hinge Loss Formulation**:
  $$\min_{\mathbf{w}} \frac{\lambda}{2} \|\mathbf{w}\|_2^2 + \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$

### 2.3 Kernel SVM & Dual Formulation
- Via Lagrangian duality:
  $$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \; \sum_{i=1}^N \alpha_i y_i = 0$$
- **Common Kernels**:
  - Linear: $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$
  - Polynomial: $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + c)^d$
  - Radial Basis Function (RBF / Gaussian): $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|_2^2)$
  - Sigmoid: $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + c)$

### 2.4 Support Vector Regression (SVR)
- Uses $\epsilon$-insensitive loss function: points within $|y - \hat{y}| \le \epsilon$ incur zero penalty.

---

## 3. Instance-Based Learning: k-Nearest Neighbors (k-NN)

- **Mechanics**: Non-parametric, lazy learner; computes distance from query point $\mathbf{x}_q$ to all training samples and aggregates the $k$ closest neighbors.
- **Distance Metrics**:
  - Euclidean: $d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_j (x_j - z_j)^2}$
  - Manhattan ($L_1$): $d(\mathbf{x}, \mathbf{z}) = \sum_j |x_j - z_j|$
  - Minkowski ($L_p$): $d(\mathbf{x}, \mathbf{z}) = \left(\sum_j |x_j - z_j|^p\right)^{1/p}$
  - Cosine Distance: $1 - \frac{\mathbf{x} \cdot \mathbf{z}}{\|\mathbf{x}\|_2 \|\mathbf{z}\|_2}$
- **Prediction Aggregation**:
  - Classification: Majority vote or inverse-distance weighted vote: $w_i = \frac{1}{d(\mathbf{x}_q, \mathbf{x}_i) + \epsilon}$.
  - Regression: Arithmetic mean or weighted mean: $\hat{y} = \frac{\sum w_i y_i}{\sum w_i}$.
- **Accelerated Search Data Structures**:
  - KD-Tree ($O(\log N)$ in low dimensions, degrades when $D > 20$).
  - Ball Tree (metric tree effective in higher dimensions).
  - Approximate Nearest Neighbor (ANN): HNSW, ScaNN, Faiss IVF.

---

## 4. Probabilistic Graphical Models: Naive Bayes

- **Bayes' Theorem Formulation**:
  $$P(y=c|\mathbf{x}) = \frac{P(y=c) \prod_{j=1}^D P(x_j | y=c)}{P(\mathbf{x})}$$
- **Conditional Independence Assumption**: Features $x_j$ are conditionally independent given class $c$.
- **Variants**:
  - **Gaussian Naive Bayes**: Continuous features assumed normally distributed:
    $$P(x_j | y=c) = \frac{1}{\sqrt{2\pi\sigma_{cj}^2}} \exp\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$
  - **Multinomial Naive Bayes**: Discrete count data (word counts, TF); Laplace smoothing:
    $$\hat{\theta}_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha D}$$
  - **Bernoulli Naive Bayes**: Binary occurrence features ($x_j \in \{0, 1\}$).
  - **Complement Naive Bayes**: Adapts Multinomial NB for imbalanced class distributions by estimating parameters using data from all classes *except* $c$.

---

## 5. Tree-Based Models: Decision Trees

- **Architecture**: Hierarchical binary or multi-way recursive partitioning of feature space into axis-aligned hyper-rectangles $R_m$.
- **Splitting Criteria**:
  - **Classification**:
    - Gini Impurity: $I_G(p) = 1 - \sum_{k=1}^K p_k^2$
    - Shannon Entropy: $H(p) = -\sum_{k=1}^K p_k \log_2(p_k)$
    - Information Gain: $\Delta = I_{\text{parent}} - \sum \frac{N_{\text{child}}}{N_{\text{parent}}} I_{\text{child}}$
  - **Regression**:
    - Variance Reduction / Mean Squared Error: $\text{MSE}(R_m) = \frac{1}{|R_m|} \sum_{i \in R_m} (y_i - \bar{y}_m)^2$
    - Mean Absolute Error (MAE): split evaluates median deviations.
- **Algorithms**:
  - **ID3**: Information gain, categorical features only.
  - **C4.5**: Gain Ratio (normalizing for intrinsic split information), continuous features, post-pruning.
  - **CART (Classification and Regression Trees)**: Binary splits, Gini / MSE, cost-complexity pruning ($R_\alpha(T) = R(T) + \alpha |T|$).

---

## 6. Ensemble Learning

Ensemble methods combine multiple base estimators to reduce variance (Bagging) or bias (Boosting).

### 6.1 Bagging & Random Forests
- **Bootstrap Aggregating (Bagging)**: Train $B$ models on independent bootstrap samples $\mathcal{D}_b$ drawn with replacement from $\mathcal{D}$.
- **Random Forest (Breiman)**:
  - Bootstrap sampling of data instances.
  - Random feature subspace selection: at each split, only evaluate $m \ll D$ randomly chosen candidate features ($m = \sqrt{D}$ for classification, $m = D/3$ for regression).
  - Decorrelates individual trees, reducing variance from $\rho \sigma^2 + \frac{1-\rho}{B} \sigma^2 \to \frac{\sigma^2}{B}$.
- **Extremely Randomized Trees (Extra-Trees)**: Randomizes both feature candidate selection AND the cut-point threshold, further reducing variance.
- **Out-of-Bag (OOB) Estimation**: Samples not included in a tree's bootstrap sample serve as a built-in validation set.

### 6.2 Boosting Algorithms
Boosting sequentially trains weak learners $h_t(\mathbf{x})$, each correcting the residual mistakes of its predecessor.

- **AdaBoost (Adaptive Boosting)**:
  - Updates sample weights: $w_{t+1, i} = w_{t, i} \exp(-\alpha_t y_i h_t(\mathbf{x}_i))$.
  - Model weight: $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$.
  - Exponential loss minimization: $\mathcal{L}(y, f(\mathbf{x})) = e^{-y f(\mathbf{x})}$.
- **Gradient Boosted Decision Trees (GBDT)**:
  - Fits trees to the pseudo-residuals (negative gradient of the arbitrary loss function):
    $$r_{ti} = -\left[ \frac{\partial \mathcal{L}(y_i, f(\mathbf{x}_i))}{\partial f(\mathbf{x}_i)} \right]_{f = f_{t-1}}$$
  - Learning rate shrinkage: $f_t(\mathbf{x}) = f_{t-1}(\mathbf{x}) + \eta \sum \gamma_{tm} \mathbb{I}(\mathbf{x} \in R_{tm})$.
- **XGBoost (Extreme Gradient Boosting)**:
  - Second-order Taylor expansion of the loss function:
    $$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
    where $g_i = \partial_{\hat{y}} \mathcal{L}$, $h_i = \partial_{\hat{y}}^2 \mathcal{L}$.
  - Optimal leaf weights: $w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$.
  - Split Gain:
    $$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{L} g_i)^2}{\sum_{L} h_i + \lambda} + \frac{(\sum_{R} g_i)^2}{\sum_{R} h_i + \lambda} - \frac{(\sum_{I} g_i)^2}{\sum_{I} h_i + \lambda} \right] - \gamma$$
  - Exact greedy, approximate quantile, and histogram-based splitting.
- **LightGBM**:
  - GOSS (Gradient-based One-Side Sampling): Retains instances with large gradients, randomly samples small gradient instances.
  - EFB (Exclusive Feature Bundling): Combines mutually exclusive sparse features into dense features.
  - Leaf-wise (best-first) tree growth rather than level-wise (depth-first).
- **CatBoost**:
  - Target Statistics with Ordered Boosting to eliminate target leakage.
  - Symmetric / oblivious decision trees for ultra-fast GPU/CPU inference.

### 6.3 Stacking & Voting Classifiers
- **Voting Classifier**:
  - Hard voting: majority rule over predicted class labels.
  - Soft voting: argmax over averaged predicted probabilities $\frac{1}{M}\sum_{m=1}^M \hat{P}_m(y=c|\mathbf{x})$.
- **Stacking (Stacked Generalization)**:
  - Base estimators generate out-of-fold cross-validated predictions $\mathbf{Z} \in \mathbb{R}^{N \times M}$.
  - A meta-learner (e.g., Logistic Regression or Ridge) learns optimal weights to blend the predictions.

---

## 7. Time Series Forecasting

### 7.1 Statistical Classical Models
- **Autoregressive (AR($p$))**: $y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t$.
- **Moving Average (MA($q$))**: $y_t = \mu + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$.
- **ARMA($p, q$)**: Combines AR and MA processes for stationary series.
- **ARIMA($p, d, q$)**: Integrates order-$d$ differencing $\Delta^d y_t$ to achieve stationarity.
- **SARIMA($p, d, q$)$\times$($P, D, Q$)$_s$**: Incorporates seasonal differencing and seasonal lag terms.

### 7.2 Machine Learning Formulations for Time Series
- **Feature Engineering**:
  - Lag features ($y_{t-1}, y_{t-2}, \dots$).
  - Rolling window statistics (mean, std, min, max, skew over window $w$).
  - Calendar & Fourier terms ($\sin(2\pi k t / P), \cos(2\pi k t / P)$).
- **Validation**:
  - **Walk-Forward Validation / TimeSeriesSplit**: Expanding or rolling train window, strictly future test window to eliminate lookahead bias / data leakage.

---

## 8. Summary Comparison Matrix

| Algorithm Family | Interpretability | Sample Efficiency | Scalability ($N$) | Dimensionality ($D$) | Non-Linearity | Primary Hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OLS / Ridge / Lasso** | High | High | Very High | High ($L_1$/$L_2$) | Linear only | $\alpha, \lambda, \text{l1\_ratio}$ |
| **Logistic Regression** | High | High | Very High | High | Linear boundary | $C, \text{penalty}, \text{solver}$ |
| **Linear SVM** | Moderate | High | High | Very High | Linear boundary | $C, \text{loss}$ |
| **Kernel SVM (RBF)** | Low | High | Poor ($O(N^2)$ to $O(N^3)$) | Moderate | High | $C, \gamma, \text{kernel}$ |
| **k-NN** | Moderate | Moderate | Poor at inference | Poor ($D > 20$) | High | $k, \text{metric}, \text{weights}$ |
| **Naive Bayes** | High | Very High | Very High | High | Linear / Quadratic | $\alpha, \text{var\_smoothing}$ |
| **Decision Trees** | High | Moderate | High | Moderate | High | $\text{max\_depth}, \text{min\_samples\_split}$ |
| **Random Forest** | Moderate | Moderate | High | High | High | $\text{n\_estimators}, \text{max\_features}$ |
| **GBDT / XGBoost / LightGBM** | Low | High | Very High | High | High | $\eta, \text{max\_depth}, \text{subsample}, \lambda$ |
