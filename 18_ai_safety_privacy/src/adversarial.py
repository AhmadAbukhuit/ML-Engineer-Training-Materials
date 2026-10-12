"""Adversarial Perturbation Attacks: FGSM and PGD in NumPy."""

from collections.abc import Callable
import numpy as np


class FGSM:
    """Fast Gradient Sign Method (Goodfellow et al., 2014).

    Generates L_infinity bounded adversarial perturbations in a single step:
    x_adv = clip( x + eps * sign( grad_x L(theta, x, y) ), clip_min, clip_max )

    Parameters
    ----------
    epsilon : float
        Maximum L_infinity perturbation magnitude.
    clip_min : float
        Minimum valid feature value.
    clip_max : float
        Maximum valid feature value.
    """

    def __init__(self, epsilon: float = 0.1, clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        self.epsilon = float(epsilon)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def generate(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Generate adversarial sample given an input x and its loss gradient function.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (N, D) or (D,).
        grad_fn : Callable
            Function returning gradient of loss with respect to input: grad_x L.
        """
        x = np.asarray(x, dtype=np.float64)
        grad = grad_fn(x)
        perturbation = self.epsilon * np.sign(grad)
        x_adv = x + perturbation
        return np.clip(x_adv, self.clip_min, self.clip_max)


class PGD:
    """Projected Gradient Descent Adversarial Attack (Madry et al., 2018).

    Iterative multi-step L_infinity attack with random start initialization:
    x_0 = x + Uniform(-eps, eps)
    x_(t+1) = Clip_B(x, eps) [ x_t + alpha * sign(grad_x L) ]

    Parameters
    ----------
    epsilon : float
        Maximum L_infinity perturbation budget.
    alpha : float
        Step size per iteration.
    n_steps : int
        Number of attack iterations.
    random_start : bool
        Whether to initialize with random uniform perturbation within eps-ball.
    clip_min : float
        Minimum valid feature value.
    clip_max : float
        Maximum valid feature value.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: float = 0.02,
        n_steps: int = 10,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.n_steps = int(n_steps)
        self.random_start = random_start
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.random_state = random_state

    def generate(
        self,
        x: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Generate PGD adversarial examples."""
        x_orig = np.asarray(x, dtype=np.float64)
        x_adv = x_orig.copy()

        rng = np.random.default_rng(self.random_state)
        if self.random_start:
            noise = rng.uniform(-self.epsilon, self.epsilon, size=x_orig.shape)
            x_adv = np.clip(x_orig + noise, self.clip_min, self.clip_max)

        for _ in range(self.n_steps):
            grad = grad_fn(x_adv)
            # Take gradient ascent step
            x_adv = x_adv + self.alpha * np.sign(grad)

            # Project back onto L_infinity ball around original x
            diff = np.clip(x_adv - x_orig, -self.epsilon, self.epsilon)
            x_adv = np.clip(x_orig + diff, self.clip_min, self.clip_max)

        return x_adv
