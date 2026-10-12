# Complete Catalog of Recommendation Systems & Collaborative Filtering

Large-scale recommender systems operate under a classical two-stage architecture: **Candidate Generation (Retrieval)** which filters millions of items down to hundreds of candidates in sub-10ms, followed by **Ranking & Scoring** which evaluates complex non-linear feature interactions to produce the final prioritized list.

---

## 1. Candidate Retrieval: Alternating Least Squares (Implicit ALS)

Collaborative filtering with explicit ratings (e.g. 1-to-5 stars) is rarely available at scale. Real-world platforms observe **implicit feedback** (clicks, purchases, watch duration, impressions), where absence of engagement does not necessarily imply dislike.

### 1.1 The Hu-Koren-Volinsky Implicit Formulation

Given raw interaction counts $r_{ui} \ge 0$, define binary preference $p_{ui}$ and confidence level $c_{ui}$:

$$p_{ui} = \begin{cases} 1 & \text{if } r_{ui} > 0 \\ 0 & \text{if } r_{ui} = 0 \end{cases}, \quad c_{ui} = 1 + \alpha r_{ui}$$

where $\alpha$ is a confidence scaling hyperparameter (typically $\alpha \in [10, 40]$).

The loss function minimizes weighted squared error with L2 regularization:

$$\mathcal{L}(X, Y) = \sum_{u, i} c_{ui} (p_{ui} - x_u^T y_i)^2 + \lambda \left( \sum_u \|x_u\|_2^2 + \sum_i \|y_i\|_2^2 \right)$$

### 1.2 Alternating Optimization

While jointly non-convex, fixing $Y$ makes the problem quadratic in $x_u$, permitting closed-form updates:

$$x_u = \left( Y^T C^u Y + \lambda I \right)^{-1} Y^T C^u p_u$$

Because $C^u = I + (C^u - I)$ where $C^u - I$ is sparse (non-zero only for items user $u$ interacted with), this can be computed efficiently:

$$Y^T C^u Y = Y^T Y + Y_{\text{active}}^T (C^u_{\text{active}} - I) Y_{\text{active}}$$

Precomputing $Y^T Y \in \mathbb{R}^{f \times f}$ once per epoch enables solving for each user in $O(f^2 n_u + f^3)$ time, where $n_u \ll |I|$.

---

## 2. Deep Candidate Retrieval: Two-Tower Dual Encoder Architectures

To incorporate rich user context (demographics, recent search queries) and item metadata (categories, tags, text embeddings), modern retrieval leverages Two-Tower neural networks.

### 2.1 Architecture

- **User Tower:** $u = f_\theta(x_{\text{user}}) \in \mathbb{R}^d$
- **Item Tower:** $v = g_\phi(x_{\text{item}}) \in \mathbb{R}^d$
- **Normalization:** $\hat{u} = \frac{u}{\|u\|_2}$, $\hat{v} = \frac{v}{\|v\|_2}$
- **Affinity Score:** $s(u, v) = \langle \hat{u}, \hat{v} \rangle = \cos(u, v)$

### 2.2 In-Batch Contrastive Training

Given a mini-batch of $B$ positive user-item interactions $\{(u_b, v_b)\}_{b=1}^B$:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\langle \hat{u}_i, \hat{v}_i \rangle / \tau)}{\sum_{j=1}^B \exp(\langle \hat{u}_i, \hat{v}_j \rangle / \tau)}$$

where $\tau > 0$ is the temperature parameter.

- **Inference Efficiency:** Item tower representations $\hat{v}_j$ are precomputed offline and indexed in approximate nearest neighbor (ANN) vector databases (e.g. HNSW, ScaNN, Faiss), enabling sub-5ms retrieval over billions of items via Maximum Inner Product Search (MIPS).

---

## 3. Second-Order Ranking: Factorization Machines (FM)

Once candidates are retrieved, a heavy ranker predicts Click-Through Rate (CTR) $\hat{y} = P(Y=1 | x)$. Standard linear regression ignores feature interactions, while polynomial regression suffers from $O(d^2)$ computational complexity and catastrophic sparsity issues when features are one-hot encoded categorical variables.

### 3.1 The Factorization Machine Formulation (Rendle, 2010)

Instead of estimating an independent interaction weight $w_{ij}$ for every pair $(i, j)$, Factorization Machines factorize the interaction matrix using low-rank latent vectors $v_i \in \mathbb{R}^k$:

$$\hat{y}(x) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle v_i, v_j \rangle x_i x_j$$

### 3.2 Linear-Time Interaction Calculation

The double summation over all pairs can be reformulated algebraically in $O(k \cdot d)$ linear time:

$$\sum_{i=1}^d \sum_{j=i+1}^d \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^d v_{i, f} x_i \right)^2 - \sum_{i=1}^d v_{i, f}^2 x_i^2 \right]$$

This mathematical equivalence allows computing complex second-order feature interactions on ultra-sparse, million-dimensional feature vectors with minimal computational overhead.
