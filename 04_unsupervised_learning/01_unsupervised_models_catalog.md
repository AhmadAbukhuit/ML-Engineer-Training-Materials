# Complete Catalog of Unsupervised Learning Models

Unsupervised Learning is the paradigm where an algorithm extracts structure, probability distributions, latent representations, or clusters from unlabeled data $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N \subset \mathbb{R}^D$ without ground-truth supervisory signals $y_i$.

$$\max_{\boldsymbol{\theta}} \sum_{i=1}^N \log p(\mathbf{x}_i; \boldsymbol{\theta}) \quad \text{or} \quad \min_{\mathcal{C}, \boldsymbol{\mu}} \sum_{k=1}^K \sum_{\mathbf{x} \in \mathcal{C}_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

---

## 1. Centroid-Based & Partitioning Clustering

### 1.1 K-Means Clustering
- **Optimization Objective (Inertia / WCSS)**:
  $$J(\mathcal{C}, \boldsymbol{\mu}) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in \mathcal{C}_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$
- **Lloyd's Algorithm (Alternating Minimization)**:
  1. *Assignment Step*: Assign each point to its nearest centroid:
     $$\mathcal{C}_k^{(t)} = \left\{ \mathbf{x}_i : \|\mathbf{x}_i - \boldsymbol{\mu}_k^{(t)}\|^2 \le \|\mathbf{x}_i - \boldsymbol{\mu}_j^{(t)}\|^2, \; \forall j \right\}$$
  2. *Update Step*: Recompute centroid as arithmetic mean of assigned points:
     $$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|\mathcal{C}_k^{(t)}|} \sum_{\mathbf{x}_i \in \mathcal{C}_k^{(t)}} \mathbf{x}_i$$
- **Convergence**: Guaranteed to converge to a local minimum since each step monotonically decreases $J$; NP-hard in general.

### 1.2 K-Means++ Initialization
- **Problem**: Random centroid initialization frequently converges to suboptimal local minima.
- **Algorithm (Arthur & Vassilvitskii, 2007)**:
  1. Choose first center $\boldsymbol{\mu}_1$ uniformly at random from $\mathcal{D}$.
  2. For remaining $K-1$ centers, compute shortest distance $D(\mathbf{x}) = \min_{j} \|\mathbf{x} - \boldsymbol{\mu}_j\|_2$.
  3. Sample next center with probability proportional to squared distance:
     $$P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}' \in \mathcal{D}} D(\mathbf{x}')^2}$$
- **Theoretical Bound**: Provides an $O(\log K)$-competitive approximation to the optimal clustering.

### 1.3 Mini-Batch K-Means
- Samples mini-batches of size $b \ll N$ at each iteration.
- Computes convex combination updates: $\boldsymbol{\mu}_k \leftarrow (1 - \eta) \boldsymbol{\mu}_k + \eta \mathbf{x}_i$, enabling $O(b)$ complexity per step and scaling to billions of data points.

### 1.4 K-Medoids (PAM - Partitioning Around Medoids)
- Centroids are restricted to actual observed data points (medoids).
- Minimizes arbitrary distance metrics (e.g., Manhattan $L_1$ or custom dissimilarity matrices), offering extreme robustness to outliers compared to K-Means.

---

## 2. Density-Based Clustering

### 2.1 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Core Definitions**:
  - $\epsilon$-Neighborhood: $N_\epsilon(\mathbf{x}) = \{\mathbf{z} \in \mathcal{D} : \|\mathbf{x} - \mathbf{z}\| \le \epsilon\}$.
  - **Core Point**: Point $\mathbf{x}$ such that $|N_\epsilon(\mathbf{x})| \ge \text{MinPts}$.
  - **Directly Density-Reachable**: $\mathbf{z} \in N_\epsilon(\mathbf{x})$ where $\mathbf{x}$ is a core point.
  - **Density-Reachable**: Chain of directly density-reachable points connecting $\mathbf{p}$ to $\mathbf{q}$.
  - **Density-Connected**: Two points $\mathbf{p}$ and $\mathbf{q}$ both density-reachable from a common core point $\mathbf{o}$.
  - **Noise / Outliers**: Points that are neither core points nor density-reachable from any core point.
- **Strengths**: Arbitrary cluster shapes (non-spherical, rings, spirals); automatically detects number of clusters; isolates noise points.
- **Weaknesses**: Sensitive to choice of $\epsilon$ and $\text{MinPts}$; struggles when clusters have varying densities.

### 2.2 HDBSCAN (Hierarchical DBSCAN)
- Replaces fixed $\epsilon$ by constructing a cluster tree across all possible $\epsilon$ scales using mutual reachability distance:
  $$d_{\text{mreach}-k}(\mathbf{a}, \mathbf{b}) = \max(\text{core}_k(\mathbf{a}), \text{core}_k(\mathbf{b}), d(\mathbf{a}, \mathbf{b}))$$
- Condenses the hierarchy and selects flat clusters based on cluster lifetime and stability ($\int (\lambda_{\text{death}} - \lambda_{\text{birth}}) \, d\lambda$).

### 2.3 OPTICS (Ordering Points To Identify the Clustering Structure)
- Generates an augmented ordering of the dataset representing its density-based clustering structure via core distance and reachability distance.

---

## 3. Distribution-Based / Probabilistic Clustering

### 3.1 Gaussian Mixture Models (GMM)
- **Generative Process**: Assumes data generated from a mixture of $K$ multivariate Gaussian distributions:
  $$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \quad \sum_{k=1}^K \pi_k = 1, \; \pi_k \ge 0$$
- **Expectation-Maximization (EM) Algorithm**:
  - **E-Step (Expectation)**: Compute posterior responsibility that component $k$ generated sample $\mathbf{x}_i$:
    $$\gamma_{ik} = P(z_i = k | \mathbf{x}_i) = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$
  - **M-Step (Maximization)**: Recompute parameters:
    $$N_k = \sum_{i=1}^N \gamma_{ik}, \quad \pi_k = \frac{N_k}{N}, \quad \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} \mathbf{x}_i$$
    $$\boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$
- **Covariance Constraints**: Full (arbitrary ellipsoids), Tied (shared covariance across clusters), Diagonal (axis-aligned ellipsoids), Spherical (equal variance along all axes).
- **Model Selection**: Evaluated using Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC):
  $$\text{BIC} = -2 \ln \hat{L} + k \ln(N)$$

---

## 4. Hierarchical & Graph Clustering

### 4.1 Agglomerative Hierarchical Clustering
- Bottom-up paradigm: Begins with $N$ singleton clusters and iteratively merges the closest pair according to a linkage criterion:
  - **Single Linkage (Minimum)**: $d(A, B) = \min_{\mathbf{a} \in A, \mathbf{b} \in B} d(\mathbf{a}, \mathbf{b})$ (prone to chaining).
  - **Complete Linkage (Maximum)**: $d(A, B) = \max_{\mathbf{a} \in A, \mathbf{b} \in B} d(\mathbf{a}, \mathbf{b})$ (favors compact clusters).
  - **Average Linkage (UPGMA)**: $d(A, B) = \frac{1}{|A||B|} \sum_{\mathbf{a} \in A} \sum_{\mathbf{b} \in B} d(\mathbf{a}, \mathbf{b})$.
  - **Ward's Minimum Variance Linkage**: Merges clusters that minimize the increase in total within-cluster variance:
    $$\Delta \text{ESS} = \frac{|A||B|}{|A| + |B|} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|_2^2$$
- Visualized as a **Dendrogram**.

### 4.2 Spectral Clustering
- Constructs an affinity graph $W$ with adjacency/Gaussian edge weights $W_{ij} = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$.
- Computes the unnormalized graph Laplacian $L = D - W$ or normalized Laplacian $L_{\text{sym}} = D^{-1/2} L D^{-1/2}$.
- Computes eigenvectors corresponding to the $K$ smallest eigenvalues of $L$.
- Clusters the low-dimensional embedding using standard K-Means.
- Solves graph cut problems (Ratio Cut, Normalized Cut).

---

## 5. Dimensionality Reduction & Manifold Learning

### 5.1 Principal Component Analysis (PCA)
- Orthogonal linear transformation projecting data onto directions of maximal variance:
  $$\max_{\mathbf{w}} \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\|_2 = 1$$
- **SVD Formulation**:
  For centered matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, compute SVD:
  $$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
  Right singular vectors $\mathbf{V}$ are eigenvectors of empirical covariance matrix $\mathbf{X}^T \mathbf{X} / (N-1)$.
- **Explained Variance Ratio**: $\lambda_k / \sum_{j=1}^D \lambda_j = \sigma_k^2 / \sum \sigma_j^2$.
- **Extensions**:
  - **Kernel PCA**: Uses Mercer kernels $K(\mathbf{x}_i, \mathbf{x}_j)$ to project non-linear manifolds into high-dimensional Hilbert space.
  - **Incremental PCA**: Mini-batch SVD for memory-constrained out-of-core data.

### 5.2 Singular Value Decomposition (SVD) & Truncated SVD
- Directly applicable to sparse matrices (e.g., TF-IDF document-term matrices in Latent Semantic Analysis / LSA) without explicit mean-centering.

### 5.3 Independent Component Analysis (ICA)
- Assumes observed signals $\mathbf{x} = \mathbf{A}\mathbf{s}$ are linear combinations of statistically independent, non-Gaussian source components $\mathbf{s}$.
- Maximizes non-Gaussianity (negentropy, kurtosis) via FastICA.

### 5.4 Manifold Learning (Non-Linear Dimensionality Reduction)
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
  - Models pairwise similarities in high-dimensional space as Gaussian probabilities $p_{j|i}$.
  - Models low-dimensional similarities using Student-t distribution $q_{ij} = (1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1} / Z$.
  - Minimizes Kullback-Leibler divergence: $D_{\text{KL}}(P \parallel Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$.
  - Ideal for 2D/3D visualization; non-parametric (cannot project new unseen points).
- **UMAP (Uniform Manifold Approximation and Projection)**:
  - Grounded in Riemannian geometry and algebraic topology.
  - Preserves both local and global structure faster than t-SNE; supports transform of out-of-sample points.
- **Isomap**: Computes geodesic distances on neighborhood graph via Dijkstra/Floyd-Warshall, then applies classical Multidimensional Scaling (MDS).
- **Locally Linear Embedding (LLE)**: Represents each point as linear combination of neighbors, preserving local patch reconstructions in lower dimension.

---

## 6. Unsupervised Anomaly & Outlier Detection

### 6.1 Isolation Forest
- **Hypothesis**: Anomalies are "few and different"; they require fewer random partition splits to isolate than normal points.
- **Average Path Length in BST**:
  $$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$
- **Anomaly Score**:
  $$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}[h(\mathbf{x})]}{c(n)}}$$
  - $s \to 1$: Definite anomaly.
  - $s < 0.5$: Normal instance.
- **Computational Complexity**: $O(t \cdot \psi \log \psi)$ where $\psi$ is subsample size (e.g., 256) and $t$ is number of trees. Linear time complexity.

### 6.2 Local Outlier Factor (LOF)
- Measures local density deviation of an object relative to its $k$-nearest neighbors.
- Reachability distance: $\text{reach-dist}_k(\mathbf{p}, \mathbf{o}) = \max(k\text{-distance}(\mathbf{o}), d(\mathbf{p}, \mathbf{o}))$.
- Outlier score $\text{LOF}_k(\mathbf{p}) \gg 1$ indicates lower local density than neighbors.

### 6.3 One-Class Support Vector Machine (OC-SVM)
- Fits a hyperplane separating all training data from the origin in feature space with maximum margin:
  $$\min_{\mathbf{w}, \boldsymbol{\xi}, \rho} \frac{1}{2} \|\mathbf{w}\|^2 + \frac{1}{\nu N} \sum_{i=1}^N \xi_i - \rho \quad \text{s.t.} \quad \langle \mathbf{w}, \Phi(\mathbf{x}_i) \rangle \ge \rho - \xi_i$$
- Parameter $\nu \in (0, 1]$ bounds the fraction of outliers.

### 6.4 Minimum Covariance Determinant (Elliptic Envelope)
- Assumes inlier data follows Gaussian distribution; robustly estimates covariance matrix by finding subset of $h$ observations whose covariance has minimum determinant.
- Anomaly score via Mahalanobis distance: $D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$.

---

## 7. Association Rule Mining

### 7.1 Key Metrics
- **Support**: $\text{supp}(X \to Y) = P(X \cup Y) = \frac{\text{count}(X \cup Y)}{N}$.
- **Confidence**: $\text{conf}(X \to Y) = P(Y | X) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)}$.
- **Lift**: $\text{lift}(X \to Y) = \frac{P(X \cup Y)}{P(X)P(Y)} = \frac{\text{conf}(X \to Y)}{\text{supp}(Y)}$. Lift $> 1$ indicates positive dependency.

### 7.2 Algorithms
- **Apriori**: Exploits downward-closure property (all subsets of a frequent itemset must be frequent) for level-wise candidate generation and pruning.
- **FP-Growth (Frequent Pattern Growth)**: Compresses transactions into a compact Frequent Pattern Tree (FP-Tree), eliminating candidate generation and mining frequent itemsets directly via recursive conditional FP-Trees.
- **Eclat (Equivalence Class Clustering and Transformation)**: Uses vertical transaction ID (TID) sets; determines support by set intersections.

---

## 8. Summary Comparison Matrix

| Algorithm | Paradigm | Primary Objective | Key Hyperparameters | Complexity | Strengths / Best Fit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Means / K-Means++** | Partitioning | Minimize within-cluster variance | $K$, max_iter, tol | $O(N K D)$ | Fast, spherical clusters |
| **DBSCAN** | Density-based | Density reachability | $\epsilon$, MinPts | $O(N \log N)$ to $O(N^2)$ | Arbitrary shapes, noise detection |
| **Gaussian Mixture (GMM)** | Probabilistic | Maximize log-likelihood via EM | $K$, covariance_type | $O(N K D^2)$ | Soft clustering, ellipsoidal distributions |
| **Agglomerative (Ward)** | Hierarchical | Minimize merge variance increase | linkage, metric | $O(N^2 \log N)$ | Interpretability, tree structure |
| **PCA** | Linear Projection | Maximize variance / minimize reconstruction error | $n\_components$ | $O(\min(N D^2, N^2 D))$ | Fast, decorrelation, linear features |
| **t-SNE / UMAP** | Manifold Learning | Preserve neighborhood topology | perplexity, n_neighbors | $O(N \log N)$ | Non-linear 2D/3D visualization |
| **Isolation Forest** | Tree Ensemble | Partition depth anomaly scoring | n_estimators, contamination | $O(N \cdot t \log \psi)$ | High-dimensional anomaly detection |
