# Supervised Learning: From Linear Baselines to Gradient Boosted Trees

Supervised machine learning algorithms learn a mapping function $f: \mathcal{X} \to \mathcal{Y}$ from labeled training pairs $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$. This guide details the mathematical principles and practical trade-offs behind foundational supervised algorithms.

---

## 1. The Bias-Variance Tradeoff

Every supervised model's expected generalization error decomposes into three distinct components:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}[\hat{f}(\mathbf{x})]^2 + \text{Var}[\hat{f}(\mathbf{x})] + \sigma^2$$

* **Bias ($\text{Bias}[\hat{f}(\mathbf{x})]^2$):** Error from erroneous assumptions in the learning algorithm. High bias causes **underfitting** (e.g., trying to fit a complex non-linear curve with a straight line).
* **Variance ($\text{Var}[\hat{f}(\mathbf{x})]$):** Error from sensitivity to small fluctuations in the training set. High variance causes **overfitting** (e.g., a deep, unconstrained decision tree memorizing training noise).
* **Irreducible Error ($\sigma^2$):** Noise inherent in the problem itself (sensor noise, unmeasured latent variables).

| Model Class | Bias Tendency | Variance Tendency | Counter-Strategy |
| :--- | :--- | :--- | :--- |
| **Linear / Logistic Regression** | High | Low | Feature engineering, non-linear kernels, polynomial terms |
| **Deep Decision Trees** | Low | High | Pruning, limiting `max_depth`, ensemble bagging (Random Forests) |
| **Ensembles (Random Forest)** | Low | Moderately Low | Aggregating predictions across decorrelated trees |
| **Gradient Boosting (XGBoost)** | Tunable | Tunable | Early stopping, learning rate shrinkage ($\eta$), tree regularization |

---

## 2. Linear Regression & Regularization

Linear regression models the relationship between continuous dependent variable $y \in \mathbb{R}$ and independent vector $\mathbf{x} \in \mathbb{R}^D$:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

### Closed-Form Normal Equation
Minimizing the sum of squared residuals leads directly to the analytical solution:

$$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

* **Computational Complexity:** Computing $(X^T X)^{-1}$ takes $\mathcal{O}(D^3)$. If the feature count $D > 10,000$, matrix inversion becomes computationally intractable and gradient descent must be used instead.
* **Multicollinearity:** If features are linearly dependent, $X^T X$ is singular (non-invertible). Regularization fixes this.

### Ridge (L2) vs. Lasso (L1) Regularization

* **Ridge Regression (L2):** Adds $\frac{\lambda}{2} \Vert{}\mathbf{w}\Vert{}_2^2$ penalty. Guarantees $X^T X + \lambda I$ is strictly invertible. Shrinks weights uniformly toward zero without setting them to exactly zero.
* **Lasso Regression (L1):** Adds $\lambda \Vert{}\mathbf{w}\Vert{}_1$ penalty. Because the L1 ball has sharp diamond corners, gradient paths hit the axes, setting coefficients to **strictly zero**. This acts as an automated feature selector.

---

## 3. Logistic Regression & Cross-Entropy

For binary classification $y \in \{0, 1\}$, we map real-valued scores to probabilities using the **Sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The log-odds (logit) is linear:

$$\ln\left(\frac{P(Y=1 \mid \mathbf{x})}{1 - P(Y=1 \mid \mathbf{x})}\right) = \mathbf{w}^T \mathbf{x} + b$$

### Maximum Likelihood & Gradient Derivation
Assuming Bernoulli-distributed labels, maximizing the log-likelihood produces the Binary Cross-Entropy loss:

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

The analytical gradient with respect to $\mathbf{w}$ is elegant and identical in form to linear regression:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{N} X^T (\hat{\mathbf{y}} - \mathbf{y})$$

---

## 4. Decision Trees: Splitting Criteria

A Decision Tree partitions the feature space into hyper-rectangles using recursive binary splitting.

### Gini Impurity vs. Shannon Entropy

* **Gini Impurity:**
  $$I_G(S) = 1 - \sum_{k=1}^C p_k^2$$
  * Measures the probability that a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.
  * Computationally faster (no logarithm required).

* **Entropy (Information Theory):**
  $$H(S) = -\sum_{k=1}^C p_k \log_2(p_k)$$
  * Measures average information / disorder.

At each internal node, the algorithm iterates over all features $j$ and possible thresholds $t$, selecting $(j^*, t^*)$ that maximizes the **Information Gain**:

$$\Delta I = I(S) - \left( \frac{|S_L|}{|S|} I(S_L) + \frac{|S_R|}{|S|} I(S_R) \right)$$

---

## 5. Modern Ensembles: Bagging vs. Boosting

### Random Forest (Bagging)
* **Principle:** Reduces **variance** by training $B$ deep, unconstrained trees in parallel.
* **Two Sources of Randomness:**
  1. **Bootstrap Aggregation:** Each tree is trained on a random sample of size $N$ drawn with replacement from the original dataset ($\sim 63.2\%$ unique samples).
  2. **Feature Subsampling:** At each split, only a random subset of $\sqrt{D}$ features is evaluated. This prevents strong predictors from dominating all trees, creating diverse, decorrelated learners.

### Gradient Boosting & XGBoost
* **Principle:** Reduces **bias** (and variance) by training shallow trees sequentially.
* **Mechanism:** Rather than training on labels directly, tree $m$ is trained to predict the **residuals** (negative gradients of the loss) of model $m-1$:
  $$\hat{y}_i^{(m)} = \hat{y}_i^{(m-1)} + \eta \cdot h_m(\mathbf{x}_i)$$
  Where $\eta \in (0, 1]$ is the shrinkage factor (learning rate).
* **XGBoost Improvements:** Uses a 2nd-order Taylor expansion of the loss (calculating both gradients $g_i$ and Hessians $h_i$), exact/approximate quantile split-finding, and built-in L1/L2 leaf weight regularization.
