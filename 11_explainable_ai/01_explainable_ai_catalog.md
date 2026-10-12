# Complete Catalog of Explainable AI (XAI) & Model Interpretability

Explainable Artificial Intelligence (XAI) encompasses the mathematical frameworks, algorithms, and methodologies designed to render the predictions of complex, non-linear machine learning models transparent, interpretable, and auditable to human practitioners.

---

## 1. Game-Theoretic Foundations: Shapley Values & SHAP

### 1.1 Cooperative Game Theory

In a cooperative game with $N$ players, a characteristic function $v(S)$ maps each subset of players $S \subseteq N$ to a real value representing the total payout achieved by that coalition, with $v(\emptyset) = 0$.
The **Shapley Value** (Lloyd Shapley, 1953) uniquely allocates the total payout among individual players fairly according to their marginal contributions:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

### 1.2 The Four Fundamental Shapley Axioms

Shapley values are the **only** payout allocation mechanism that simultaneously satisfies:

1. **Efficiency (Additivity to Total)**:
   $$\sum_{i \in N} \phi_i(v) = v(N) - v(\emptyset)$$
   The sum of all feature attributions equals the difference between the model's prediction and the expected background prediction.
2. **Symmetry**: If $v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S \subseteq N \setminus \{i, j\}$, then $\phi_i(v) = \phi_j(v)$. Identical contributors receive equal attribution.
3. **Dummy Player (Null Player)**: If $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i(v) = 0$. Features providing zero marginal gain receive zero attribution.
4. **Additivity**: For independent games $u$ and $v$, $\phi_i(u + v) = \phi_i(u) + \phi_i(v)$.

### 1.3 SHAP (SHapley Additive exPlanations - Lundberg & Lee, 2017)

- **KernelSHAP**: Formulates Shapley estimation as a weighted linear surrogate regression problem using the Shapley kernel:
  $$\pi_x(z') = \frac{|N| - 1}{\binom{|N|}{|z'|} |z'| (|N| - |z'|)}$$
- **TreeSHAP**: Explores tree structures in $O(T L D^2)$ time rather than exponential $O(2^D)$, pushing marginal expectations recursively down split paths.

---

## 2. Local Surrogate Explanations: LIME

### 2.1 Theoretical Framework (Ribeiro et al., 2016)

While complex black-box functions $f(x)$ are globally non-linear, they are approximately linear in the immediate local vicinity of an individual instance $x$.
LIME solves the constrained optimization problem:

$$\xi(x) = \arg\min_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

- $g \in \mathcal{G}$: An interpretable local surrogate model (e.g., Ridge linear regression $g(z') = w^T z'$).
- $\Omega(g)$: Complexity penalty bounding the number of active features.
- $\pi_x(z)$: Proximity measure weighting perturbed sample $z$ by distance from $x$:
  $$\pi_x(z) = \exp\left( -\frac{\|x - z\|_2^2}{\sigma^2} \right)$$
- $\mathcal{L}$: Locally weighted squared loss:
  $$\mathcal{L}(f, g, \pi_x) = \sum_{z, z' \in \mathcal{Z}} \pi_x(z) \left( f(z) - g(z') \right)^2$$

### 2.2 Strengths & Limitations

- **Strengths**: Model-agnostic, works on tabular, image (super-pixel perturbations), and text inputs.
- **Limitations**: Sampling variance (explanations can fluctuate between runs), sensitive to bandwidth parameter $\sigma$.

---

## 3. Global Feature Importance Methods

### 3.1 Mean Decrease in Impurity (MDI - Tree Default)

- Evaluates total reduction in split criterion (Gini or MSE) accumulated by feature $j$ across all ensemble trees.
- **Severe Flaw**: Strongly biased towards high-cardinality continuous features or random numerical noise; completely ignores out-of-sample generalization.

### 3.2 Permutation Feature Importance (Breiman, 2001)

- Evaluates feature importance directly on a held-out validation set $\mathcal{D}_{\text{val}}$:
  1. Record baseline evaluation metric $s_{\text{base}} = \mathcal{M}(y, \hat{f}(X))$.
  2. For feature column $j$, randomly shuffle its values across rows, creating $\tilde{X}^{(j)}$.
  3. Re-evaluate the metric: $s_{\text{perm}} = \mathcal{M}(y, \hat{f}(\tilde{X}^{(j)}))$.
  4. Feature importance is the drop in performance:
     $$I(j) = s_{\text{base}} - s_{\text{perm}}$$
- **Strengths**: True reflection of out-of-sample utility; uncorrupted by feature cardinality.

---

## 4. Marginal Feature Effects: PDP & ICE

### 4.1 Partial Dependence Plots (PDP - Friedman, 2001)

- Shows the marginal effect of one or two features $S$ on the predicted outcome:
  $$\hat{f}_S(x_S) = \frac{1}{N} \sum_{i=1}^N f(x_S, x_{i, C})$$
  where $C$ denotes the complement feature set ($N \setminus S$).
- Assumes features in $S$ and $C$ are not strongly correlated (otherwise evaluates unrealistic feature combinations).

### 4.2 Individual Conditional Expectation (ICE)

- Plots a separate curve for each individual sample $i \in \{1, \dots, N\}$ across the grid of $x_S$:
  $$\hat{f}^{(i)}_S(x_S) = f(x_S, x_{i, C})$$
- Uncovers non-linear interactions and heterogeneous effects that the averaged PDP curve washes out.

---

## 5. Summary Comparison Matrix

| Technique | Scope | Model Agnostic? | Theoretical Foundation | Computational Cost | Best Fit Scenario |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact SHAP** | Local & Global | Yes | Cooperative Game Theory | $O(2^D)$ | Small feature counts ($D \le 12$) |
| **TreeSHAP** | Local & Global | No (Trees only) | Tree Structure Recursion | Polynomial $O(T L D^2)$ | GBDT / XGBoost / LightGBM |
| **LIME** | Local | Yes | Local Surrogate Perturbation | Low $O(M D)$ | Single prediction audit |
| **Permutation Importance** | Global | Yes | Out-of-sample metric drop | $O(R \cdot D \cdot N)$ | Global feature ranking on test sets |
| **PDP / ICE** | Global / Local | Yes | Marginal Expectation | $O(K \cdot N)$ | Visualizing functional shape & non-linearity |
