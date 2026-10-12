# Complete Catalog of AI Safety, Privacy & Algorithmic Fairness

Deploying machine learning models in production exposes organizations to adversarial manipulation, data leakage, and discriminatory societal harms. Robust engineering requires proactive defense against input tampering, formal privacy protections, and mathematical bias audits.

---

## 1. Adversarial Robustness: First-Order Perturbation Attacks

Deep neural networks exhibit extreme sensitivity to small, worst-case input perturbations crafted to maximize loss while remaining imperceptible.

### 1.1 Fast Gradient Sign Method (FGSM - Goodfellow et al., 2014)
Linearizing the cost function $\mathcal{L}(\theta, x, y)$ around $x$, the perturbation that maximizes loss under an $L_\infty$ constraint $\|\delta\|_\infty \le \epsilon$ is:

$$x_{\text{adv}} = \text{clip}\left( x + \epsilon \cdot \text{sign}\left( \nabla_x \mathcal{L}(\theta, x, y) \right), \, x_{\min}, \, x_{\max} \right)$$

FGSM is a computationally cheap single-step attack, useful for adversarial training.

### 1.2 Projected Gradient Descent (PGD - Madry et al., 2018)
PGD is the standard first-order adversary, treating attack generation as constrained non-convex optimization:

$$x^{0} = x + \mathcal{U}(-\epsilon, \epsilon)$$
$$x^{t+1} = \Pi_{B_\epsilon(x)} \left[ x^t + \alpha \cdot \text{sign}\left( \nabla_{x^t} \mathcal{L}(\theta, x^t, y) \right) \right]$$

where $\Pi_{B_\epsilon(x)}$ projects back onto the $L_\infty$ $\epsilon$-ball centered at original input $x$, and $\alpha$ is the step size.

---

## 2. Differential Privacy: DP-SGD (Abadi et al., 2016)

A randomized algorithm $\mathcal{M}$ satisfies $(\epsilon, \delta)$-differential privacy if for all adjacent datasets $D, D'$ differing by a single individual:

$$\mathbb{P}(\mathcal{M}(D) \in S) \le e^\epsilon \cdot \mathbb{P}(\mathcal{M}(D') \in S) + \delta$$

### 2.1 Differentially Private Stochastic Gradient Descent
To bound an individual's influence on model weights, standard SGD is modified:
1. **Per-Sample Gradient Clipping:**
   $$\bar{g}(x_i) = \frac{g(x_i)}{\max\left(1, \, \frac{\|g(x_i)\|_2}{C}\right)}$$
   This guarantees that the $L_2$ sensitivity of the sum $\sum_i \bar{g}(x_i)$ is bounded by $C$.
2. **Gaussian Noise Injection:**
   $$\tilde{g} = \frac{1}{B} \left( \sum_{i=1}^B \bar{g}(x_i) + \mathcal{N}\left(0, \, (\sigma C)^2 I\right) \right)$$
   where $\sigma$ is the noise multiplier and $B$ is the batch size.

Privacy budgets $(\epsilon, \delta)$ accumulate across training iterations and are tracked using Rényi Differential Privacy (RDP) or the moments accountant.

---

## 3. Algorithmic Fairness & Bias Auditing

Given sensitive / protected demographic attribute $A \in \{0, 1\}$, true label $Y \in \{0, 1\}$, and model prediction $\hat{Y} \in \{0, 1\}$:

### 3.1 Demographic Parity (Statistical Parity)
The probability of a positive outcome must be identical across protected groups:

$$\mathbb{P}(\hat{Y} = 1 \mid A = 0) = \mathbb{P}(\hat{Y} = 1 \mid A = 1)$$

- **Disparate Impact Ratio:**
  $$\text{DI} = \frac{\min(\mathbb{P}(\hat{Y}=1|A=0), \mathbb{P}(\hat{Y}=1|A=1))}{\max(\mathbb{P}(\hat{Y}=1|A=0), \mathbb{P}(\hat{Y}=1|A=1))}$$
  A ratio below $0.80$ violates the US EEOC Four-Fifths Rule.

### 3.2 Equal Opportunity (Hardt et al., 2016)
The True Positive Rate (TPR) must be equal across protected groups:

$$\mathbb{P}(\hat{Y} = 1 \mid A = 0, Y = 1) = \mathbb{P}(\hat{Y} = 1 \mid A = 1, Y = 1)$$

Protects qualified individuals from discriminatory negative classifications.

### 3.3 Equalized Odds
Requires both True Positive Rate (TPR) and False Positive Rate (FPR) parity across groups:

$$\mathbb{P}(\hat{Y} = 1 \mid A = 0, Y = y) = \mathbb{P}(\hat{Y} = 1 \mid A = 1, Y = y), \quad \forall y \in \{0, 1\}$$
