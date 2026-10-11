# Phase 4: Unsupervised Learning

Welcome to the **Unsupervised Learning** phase of the Machine Learning Engineering curriculum. This phase covers clustering algorithms, density estimation, manifold learning, dimensionality reduction, anomaly detection, and pattern discovery from unlabelled data.

---

## 📚 Curriculum & Reference Guides

- [`01_unsupervised_models_catalog.md`](file:///home/ahmad/ML-Engineer-Training-Materials/04_unsupervised_learning/01_unsupervised_models_catalog.md): Comprehensive mathematical catalog of all unsupervised learning models (K-Means, K-Means++, Mini-Batch, K-Medoids, DBSCAN, HDBSCAN, OPTICS, GMM with Expectation-Maximization, Agglomerative Hierarchical, Spectral Clustering, PCA via SVD, ICA, t-SNE, UMAP, Isolation Forest, Local Outlier Factor, One-Class SVM, and Apriori / FP-Growth).

---

## 🛠️ Code Implementations (`src/`)

- [`src/kmeans.py`](file:///home/ahmad/ML-Engineer-Training-Materials/04_unsupervised_learning/src/kmeans.py): K-Means clustering with K-Means++ initialization, vectorized inertia calculation, and silhouette score.
- [`src/dbscan.py`](file:///home/ahmad/ML-Engineer-Training-Materials/04_unsupervised_learning/src/dbscan.py): Density-Based Spatial Clustering of Applications with Noise (DBSCAN) from scratch, handling core, border, and outlier points.
- [`src/gmm.py`](file:///home/ahmad/ML-Engineer-Training-Materials/04_unsupervised_learning/src/gmm.py): Gaussian Mixture Models trained via the Expectation-Maximization (EM) algorithm with numerical covariance regularization.
- [`src/pca.py`](file:///home/ahmad/ML-Engineer-Training-Materials/04_unsupervised_learning/src/pca.py): Principal Component Analysis from scratch via Singular Value Decomposition (SVD) and covariance eigendecomposition.
- [`src/isolation_forest.py`](file:///home/ahmad/ML-Engineer-Training-Materials/04_unsupervised_learning/src/isolation_forest.py): Isolation Forest ensemble for anomaly and outlier detection with average BST path length scoring.

---

## 🧪 Unit Tests (`tests/`)

Run the unsupervised test suite using pytest:

```bash
pytest 04_unsupervised_learning/tests
```
