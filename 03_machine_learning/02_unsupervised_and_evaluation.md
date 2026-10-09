# Unsupervised Learning & Rigorous Model Evaluation

In real-world ML systems, models face imbalanced distributions, missing labels, and noisy production data. This guide covers unsupervised clustering (K-Means, DBSCAN) and rigorous evaluation metrics to diagnose production performance.

---

## 1. Unsupervised Clustering

Clustering discovers natural group structures within unlabeled feature vectors $X \in \mathbb{R}^{N \times D}$.

### K-Means Algorithm (Expectation-Maximization)

K-Means partitions data into $k$ non-overlapping clusters $S = \{S_1, S_2, \dots, S_k\}$, minimizing the **Inertia** (Within-Cluster Sum of Squares):

$$\mathcal{J} = \sum_{j=1}^k \sum_{\mathbf{x} \in S_j} \Vert{}\mathbf{x} - \boldsymbol{\mu}_j\Vert{}_2^2$$

#### The Two-Step Iteration

1. **Assignment Step (E-step):** Assign each sample to the nearest centroid:
   $$S_j^{(t)} = \left\{ \mathbf{x}_i : \Vert{}\mathbf{x}_i - \boldsymbol{\mu}_j^{(t)}\Vert{}^2 \le \Vert{}\mathbf{x}_i - \boldsymbol{\mu}_l^{(t)}\Vert{}^2 \quad \forall l \right\}$$
2. **Update Step (M-step):** Recompute centroids as the center of mass of their assigned points:
   $$\boldsymbol{\mu}_j^{(t+1)} = \frac{1}{|S_j^{(t)}|} \sum_{\mathbf{x} \in S_j^{(t)}} \mathbf{x}$$

#### The K-Means++ Initialization

Standard random centroid initialization often gets trapped in catastrophic local minima. **K-Means++** solves this:

1. Pick the first centroid $\boldsymbol{\mu}_1$ uniformly at random from the dataset.
2. For each point $\mathbf{x}_i$, compute $D(\mathbf{x}_i)$, the shortest distance to any already chosen centroid.
3. Choose the next centroid with probability proportional to $D(\mathbf{x}_i)^2$:
   $$P(\boldsymbol{\mu}_{\text{new}} = \mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}'} D(\mathbf{x}')^2}$$
4. Repeat until $k$ centroids are chosen.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Unlike K-Means (which assumes convex spherical clusters and requires specifying $k$), **DBSCAN** can discover arbitrary cluster shapes and automatically identify noise/outliers:

* **$\epsilon$ (Eps):** Maximum neighborhood radius.
* **MinPts:** Minimum samples within $\epsilon$-neighborhood for a point to be considered a **Core Point**.
* **Border Points:** Within $\epsilon$ of a core point but have fewer than MinPts neighbors.
* **Noise Points:** Neither core nor border.

---

## 2. Rigorous Model Evaluation

Evaluating models with simple accuracy is misleading when working with imbalanced datasets (e.g., fraud detection with 99.9% negative transactions).

### The Confusion Matrix

| | Predicted Negative ($\hat{y}=0$) | Predicted Positive ($\hat{y}=1$) |
| :--- | :--- | :--- |
| **Actual Negative ($y=0$)** | True Negative (TN) | False Positive (FP) *(Type I Error)* |
| **Actual Positive ($y=1$)** | False Negative (FN) *(Type II Error)* | True Positive (TP) |

### Core Metrics

* **Precision (Quality of Positive Predictions):**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  * *Critical when False Positives are expensive* (e.g., email spam filter marking legitimate business emails as spam).

* **Recall / Sensitivity (Coverage of Actual Positives):**
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  * *Critical when False Negatives are catastrophic* (e.g., cancer diagnosis or fraud detection).

* **F1-Score (Harmonic Mean):**
  $$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

* **ROC Curve & AUC (Receiver Operating Characteristic):**
  * Plots True Positive Rate (Recall) vs False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$) across all possible decision thresholds $\tau \in [0, 1]$.
  * $\text{AUC} = 0.5$ represents random guessing; $\text{AUC} = 1.0$ represents a perfect classifier.

---

## 3. Resampling & Cross-Validation

To prevent data leakage and obtain unbiased variance estimates of model performance:

* **Stratified K-Fold Cross Validation:** Ensures each fold preserves the exact class ratio of the full dataset. Crucial for imbalanced targets.
* **Time-Series Split (Rolling Window):** For temporal data, NEVER use standard random shuffle cross validation (which leaks future data into the past). Always use forward chaining:
  * Fold 1: Train on $[t_1 \dots t_k]$, Test on $[t_{k+1}]$
  * Fold 2: Train on $[t_1 \dots t_{k+1}]$, Test on $[t_{k+2}]$
