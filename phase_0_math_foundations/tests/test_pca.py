import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from src.pca import PCA as CustomPCA

def test_pca_svd():
    # Generate random synthetic dataset (100 samples, 5 features)
    np.random.seed(42)
    X = np.random.rand(100, 5)
    n_components = 2
    
    # Fit Custom PCA
    custom_pca = CustomPCA(n_components=n_components)
    X_custom = custom_pca.fit_transform(X)
    
    # Fit Scikit-Learn PCA
    sklearn_pca = SklearnPCA(n_components=n_components)
    X_sklearn = sklearn_pca.fit_transform(X)
    
    # 1. Test Explained Variance Ratio
    np.testing.assert_allclose(
        custom_pca.explained_variance_ratio_, 
        sklearn_pca.explained_variance_ratio_, 
        atol=1e-7,
        err_msg="Explained variance ratio does not match."
    )
    
    # 2. Test Components (accounting for sign ambiguity)
    np.testing.assert_allclose(
        np.abs(custom_pca.components_), 
        np.abs(sklearn_pca.components_), 
        atol=1e-7,
        err_msg="Principal components do not match."
    )
    
    # 3. Test Transformed Data (accounting for sign ambiguity)
    np.testing.assert_allclose(
        np.abs(X_custom), 
        np.abs(X_sklearn), 
        atol=1e-7,
        err_msg="Transformed data projections do not match."
    )