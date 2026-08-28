import numpy as np

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) loss.
    
    Mean Squared Error (MSE) measures the average squared difference between the estimated values 
    (predictions) and the actual value. It is a risk function corresponding to the expected value 
    of the squared error loss. MSE is commonly used for regression tasks and is derived from 
    Maximum Likelihood Estimation (MLE) under the assumption that the target variable has a 
    Gaussian distribution.
    """
    return float(np.mean((y_true - y_pred) ** 2))

def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the MSE loss with respect to the predictions (y_pred).
    
    The gradient of MSE indicates the direction and magnitude of the steepest increase of the loss function.
    During optimization (e.g., gradient descent), the model parameters are updated in the opposite 
    direction of the gradient to minimize the MSE. For MSE, the gradient is proportional to the 
    prediction error (y_pred - y_true).
    """
    N = y_true.shape[0] if y_true.ndim > 0 else 1
    return 2 * (y_pred - y_true) / N

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Computes the Binary Cross-Entropy (BCE) loss.
    
    Binary Cross-Entropy (BCE) is a loss function used for binary classification tasks. It measures 
    the performance of a classification model whose output is a probability value between 0 and 1. 
    It is derived from Maximum Likelihood Estimation (MLE) assuming a Bernoulli distribution for 
    the target variable. The loss increases as the predicted probability diverges from the actual label.
    
    Predictions are clipped to [eps, 1-eps] to prevent log(0) domain errors.
    """
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)))

def bce_gradient(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Computes the gradient of BCE loss with respect to the predictions.
    
    The gradient of BCE points in the direction that maximizes the loss. By moving in the opposite 
    direction (negative gradient), a model can adjust its weights to produce probabilities closer 
    to the true binary labels. The analytical derivative simplifies to (y_pred - y_true) divided 
    by the variance of the prediction (y_pred * (1 - y_pred)).
    """
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    N = y_true.shape[0] if y_true.ndim > 0 else 1
    # Derivative of BCE is (y_pred - y_true) / (y_pred * (1 - y_pred)) averaged over N
    return (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped) * N)

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes the softmax probability distribution.
    
    Softmax is a mathematical function that converts a vector of real numbers (often called logits) 
    into a probability distribution. Every value in the output vector is bounded between 0 and 1, 
    and the sum of all values is exactly 1. It is typically used in the final layer of neural 
    networks for multi-class classification.
    
    Uses the numerical stability trick: subtracting the max value before exponentiating.
    """
    # Keepdims=True allows broadcasting the subtraction across the specified axis
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

def categorical_cross_entropy(y_true_one_hot: np.ndarray, y_pred_probs: np.ndarray, eps: float = 1e-15) -> float:
    """
    Computes the Categorical Cross-Entropy (CCE) loss.
    
    Categorical Cross-Entropy (CCE) is used for multi-class classification tasks where an example 
    belongs to exactly one class. It quantifies the difference between two probability distributions: 
    the true distribution (one-hot encoded labels) and the predicted distribution (e.g., from softmax). 
    Minimizing CCE is equivalent to maximizing the likelihood of the data under a categorical distribution.
    """
    y_pred_clipped = np.clip(y_pred_probs, eps, 1 - eps)
    # Sum over classes, then mean over the batch
    return float(-np.mean(np.sum(y_true_one_hot * np.log(y_pred_clipped), axis=-1)))

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise cosine similarity matrix between two sets of vectors A and B.
    
    Cosine Similarity measures the cosine of the angle between two non-zero vectors in an inner 
    product space. It determines how similar two vectors are, irrespective of their magnitude. 
    A value of 1 means the vectors are perfectly aligned (most similar), 0 means they are 
    orthogonal (independent), and -1 means they are diametrically opposed.
    
    A: shape (N, D)
    B: shape (M, D)
    Returns: shape (N, M)
    """
    # Ensure matrices are 2D
    if A.ndim == 1: A = A.reshape(1, -1)
    if B.ndim == 1: B = B.reshape(1, -1)

    # Compute dot products between all N x M pairs via matrix multiplication
    dot_products = np.dot(A, B.T)
    
    # Compute L2 norms along the feature dimension (D)
    norm_A = np.linalg.norm(A, axis=-1, keepdims=True)  # Shape: (N, 1)
    norm_B = np.linalg.norm(B, axis=-1, keepdims=True)  # Shape: (M, 1)
    
    # Broadcast division to compute similarities
    return dot_products / (norm_A * norm_B.T)