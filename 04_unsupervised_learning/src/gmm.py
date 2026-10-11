"""Gaussian Mixture Model (GMM) trained via Expectation-Maximization (EM) from scratch."""

import numpy as np


class GaussianMixtureModel:
    """Gaussian Mixture Model with EM optimization.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence threshold on log-likelihood difference.
    reg_covar : float
        Non-negative regularization added to the diagonal of covariance matrices.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int | None = 42,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.converged_: bool = False

    def _multivariate_gaussian_pdf(
        self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        """Compute multivariate Gaussian PDF for all samples."""
        d = X.shape[1]
        diff = X - mean
        reg_cov = cov + np.eye(d) * self.reg_covar

        # Use Cholesky for stability: cov = L L^T
        try:
            L = np.linalg.cholesky(reg_cov)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            # solve L y = diff.T -> ||y||^2
            solve_y = np.linalg.solve(L, diff.T)
            maha_sq = np.sum(solve_y**2, axis=0)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            inv_cov = np.linalg.pinv(reg_cov)
            _, log_det = np.linalg.slogdet(reg_cov)
            maha_sq = np.sum((diff @ inv_cov) * diff, axis=1)

        log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + maha_sq)
        return np.exp(np.clip(log_prob, -100, 100))

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """Fit GMM parameters using EM algorithm."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Initialize mixing weights uniformly
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

        # Randomly initialize cluster means from samples
        init_indices = rng.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[init_indices].copy()

        # Initialize covariances as empirical covariance of the dataset
        base_cov = np.cov(X, rowvar=False)
        if base_cov.ndim == 0:
            base_cov = np.array([[base_cov]])
        self.covariances_ = np.array([base_cov.copy() for _ in range(self.n_components)])

        prev_log_likelihood = -np.inf

        for _ in range(self.max_iter):
            # --- E-STEP: Expectation ---
            # Compute responsibilities gamma: shape (n_samples, n_components)
            likelihoods = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                likelihoods[:, k] = self.weights_[k] * self._multivariate_gaussian_pdf(
                    X, self.means_[k], self.covariances_[k]
                )

            total_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
            total_likelihood = np.maximum(total_likelihood, 1e-12)
            responsibilities = likelihoods / total_likelihood

            log_likelihood = np.sum(np.log(np.sum(likelihoods, axis=1) + 1e-12))

            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                break
            prev_log_likelihood = log_likelihood

            # --- M-STEP: Maximization ---
            N_k = np.sum(responsibilities, axis=0)  # effective number of samples per cluster
            N_k = np.maximum(N_k, 1e-8)

            # Update weights
            self.weights_ = N_k / n_samples

            # Update means
            self.means_ = (responsibilities.T @ X) / N_k[:, np.newaxis]

            # Update covariances
            for k in range(self.n_components):
                diff = X - self.means_[k]  # (n_samples, n_features)
                resp_k = responsibilities[:, k : k + 1]  # (n_samples, 1)
                cov_k = (resp_k * diff).T @ diff / N_k[k]
                self.covariances_[k] = cov_k + np.eye(n_features) * self.reg_covar

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability distribution over components."""
        assert self.weights_ is not None and self.means_ is not None and self.covariances_ is not None
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        likelihoods = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights_[k] * self._multivariate_gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )

        total_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
        return likelihoods / np.maximum(total_likelihood, 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign samples to highest probability component."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
