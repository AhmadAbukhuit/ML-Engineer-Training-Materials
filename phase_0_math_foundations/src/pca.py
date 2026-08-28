"""
This code implements Principal Component Analysis (PCA) from scratch.
PCA is a dimensionality reduction technique that transforms high-dimensional data 
into a lower-dimensional space while preserving as much variance (information) as possible.
While it can be computed via Covariance Eigendecomposition, this implementation uses 
Singular Value Decomposition (SVD) directly on the data matrix, which is numerically 
more stable and efficient.
"""
import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) using Singular Value Decomposition (SVD).
    """
    def __init__(self, n_components: int):
        """
        Initializes PCA with the target number of components to keep.
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        Calculates the principal components of the dataset X.
        
        Math breakdown:
        1. Centers the data by subtracting the mean of each feature.
        2. Applies SVD: X_centered = U * Sigma * V^T.
        3. The rows of V^T are the principal axes (eigenvectors).
        4. The singular values (Sigma) squared divided by N-1 give the variance explained by each axis.
        """
        n_samples, n_features = X.shape
        
        # 1. Center the data (mean normalization)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Compute SVD on the centered data matrix.
        # full_matrices=False ensures we only compute the necessary dimensions (truncated SVD)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. Extract the top k principal components (Right singular vectors from V^T)
        self.components_ = Vt[:self.n_components]
        
        # 4. Calculate explained variance for each component.
        # Variance is the square of singular values divided by degrees of freedom (N - 1)
        variance = (S ** 2) / (n_samples - 1)
        self.explained_variance_ = variance[:self.n_components]
        
        # 5. Calculate explained variance ratio (proportion of total variance explained by top k)
        total_variance = np.sum(variance)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects the input high-dimensional data onto the learned principal components,
        yielding a lower-dimensional representation.
        
        Math: X_transformed = (X - mean) * V_k
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method that fits the PCA model to the data and then transforms it 
        in a single step.
        """
        self.fit(X)
        return self.transform(X)