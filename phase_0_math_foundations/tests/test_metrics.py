import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, log_loss
from scipy.spatial.distance import cdist

# Import your implementations
from phase_0_math_foundations.src.metrics import (
    mse_loss, mse_gradient, binary_cross_entropy, bce_gradient, 
    softmax, categorical_cross_entropy, cosine_similarity_matrix
)

def test_mse_loss():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    custom_mse = mse_loss(y_true, y_pred)
    sklearn_mse = mean_squared_error(y_true, y_pred)
    
    assert np.isclose(custom_mse, sklearn_mse), "MSE Loss calculation is incorrect."

def test_mse_gradient():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([3.0, 1.0])
    
    grad = mse_gradient(y_true, y_pred)
    # Expected analytical gradient: 2 * (y_pred - y_true) / N
    # = (2/2) * [2.0, -1.0] = [2.0, -1.0]
    expected_grad = np.array([2.0, -1.0])
    
    np.testing.assert_allclose(grad, expected_grad, err_msg="MSE gradient is incorrect.")

def test_binary_cross_entropy():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    
    custom_bce = binary_cross_entropy(y_true, y_pred)
    sklearn_bce = log_loss(y_true, y_pred)
    
    assert np.isclose(custom_bce, sklearn_bce), "BCE Loss calculation is incorrect."

def test_softmax_stability():
    # Large numbers that would cause np.exp() to overflow without the stability trick
    z = np.array([[1000.0, 1001.0, 1002.0]])
    probs = softmax(z)
    
    # They should sum to 1
    assert np.isclose(np.sum(probs), 1.0)
    # The largest input should have the highest probability
    assert np.argmax(probs) == 2

def test_categorical_cross_entropy():
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    
    custom_cce = categorical_cross_entropy(y_true, y_pred)
    
    # Analytical expected: - (1*log(0.7) + 1*log(0.8)) / 2
    expected = -(np.log(0.7) + np.log(0.8)) / 2
    
    assert np.isclose(custom_cce, expected), "CCE Loss calculation is incorrect."

def test_cosine_similarity_matrix():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    
    custom_sim = cosine_similarity_matrix(A, B)
    
    # SciPy's cdist computes cosine DISTANCE (1 - similarity) [1]
    scipy_dist = cdist(A, B, metric='cosine')
    expected_sim = 1.0 - scipy_dist
    
    np.testing.assert_allclose(custom_sim, expected_sim, atol=1e-7, err_msg="Cosine similarity matrix is incorrect.")