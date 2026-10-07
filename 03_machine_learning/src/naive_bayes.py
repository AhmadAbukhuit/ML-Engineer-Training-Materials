"""
Gaussian Naive Bayes Classifier from First Principles.

This module implements Gaussian Naive Bayes for continuous features using 
Maximum Likelihood Estimation of class means, variances, and priors in pure NumPy.

Formulation (computed in log-space to avoid numerical underflow):
log P(y=c | x) = log P(y=c) + sum_i log P(x_i | y=c)
where P(x_i | y=c) ~ N(mu_{c,i}, sigma_{c,i}^2)
"""

import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None      # Shape: (n_classes, n_features)
        self.var_: np.ndarray | None = None       # Shape: (n_classes, n_features)
        self.class_prior_: np.ndarray | None = None  # Shape: (n_classes,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[idx, :] = np.mean(X_c, axis=0)
            # Add var_smoothing to avoid division by zero on zero-variance features
            self.var_[idx, :] = np.var(X_c, axis=0) + self.var_smoothing
            self.class_prior_[idx] = len(X_c) / n_samples

        return self

    def _calculate_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the log joint likelihood: log P(y=c) + sum_i log N(x_i | mu_{c,i}, var_{c,i})
        Returns array of shape (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_likelihoods = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            mean = self.mean_[idx]
            var = self.var_[idx]
            prior = self.class_prior_[idx]

            # Gaussian log PDF: -0.5 * log(2 * pi * var) - 0.5 * ((x - mean)^2 / var)
            log_prob_features = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((X - mean) ** 2 / var)
            log_likelihoods[:, idx] = np.log(prior) + np.sum(log_prob_features, axis=1)

        return log_likelihoods

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes class probabilities using the Log-Sum-Exp trick for numerical stability.
        """
        log_likes = self._calculate_log_likelihood(X)
        # Shift by max for numerical stability: exp(log_p - max) / sum(exp(log_p - max))
        max_log = np.max(log_likes, axis=1, keepdims=True)
        exp_likes = np.exp(log_likes - max_log)
        return exp_likes / np.sum(exp_likes, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_likes = self._calculate_log_likelihood(X)
        return self.classes_[np.argmax(log_likes, axis=1)]
