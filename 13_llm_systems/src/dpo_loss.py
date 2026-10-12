"""Direct Preference Optimization (DPO) Loss from First Principles in NumPy."""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    # Clip to prevent overflow
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class DPOLoss:
    """Direct Preference Optimization (DPO) Loss Evaluator (Rafailov et al., 2023).

    Eliminates explicit reward modeling in RLHF by optimizing the policy directly
    on pairwise preference pairs (prompt x, winner y_w, loser y_l).

    Parameters
    ----------
    beta : float
        Temperature scaling parameter governing deviation from the reference model (typically 0.1 to 0.5).
    """

    def __init__(self, beta: float = 0.1) -> None:
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.beta = beta

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        reference_chosen_logps: np.ndarray,
        reference_rejected_logps: np.ndarray,
    ) -> dict[str, float]:
        """Compute DPO loss, implicit rewards, and preference margins.

        Parameters
        ----------
        policy_chosen_logps : np.ndarray of shape (batch_size,)
            Log p_theta(y_w | x)
        policy_rejected_logps : np.ndarray of shape (batch_size,)
            Log p_theta(y_l | x)
        reference_chosen_logps : np.ndarray of shape (batch_size,)
            Log p_ref(y_w | x)
        reference_rejected_logps : np.ndarray of shape (batch_size,)
            Log p_ref(y_l | x)

        Returns
        -------
        dict with keys:
            'loss' : float
            'chosen_rewards' : float (mean)
            'rejected_rewards' : float (mean)
            'reward_margin' : float (mean)
            'accuracy' : float (fraction where chosen_reward > rejected_reward)
        """
        pi_chosen = np.asarray(policy_chosen_logps, dtype=np.float64)
        pi_rejected = np.asarray(policy_rejected_logps, dtype=np.float64)
        ref_chosen = np.asarray(reference_chosen_logps, dtype=np.float64)
        ref_rejected = np.asarray(reference_rejected_logps, dtype=np.float64)

        # Log ratios: log(pi / ref) = log pi - log ref
        chosen_log_ratios = pi_chosen - ref_chosen
        rejected_log_ratios = pi_rejected - ref_rejected

        # Implicit rewards r(x, y) = beta * log(pi(y|x) / ref(y|x))
        chosen_rewards = self.beta * chosen_log_ratios
        rejected_rewards = self.beta * rejected_log_ratios

        # Reward margins
        logits = chosen_rewards - rejected_rewards

        # DPO loss: -E[log sigma(logits)]
        # Stable log(sigma(z)) = -log(1 + exp(-z))
        losses = -np.log(sigmoid(logits) + 1e-12)

        loss_mean = float(np.mean(losses))
        reward_margin = float(np.mean(logits))
        accuracy = float(np.mean(chosen_rewards > rejected_rewards))

        return {
            "loss": loss_mean,
            "chosen_rewards": float(np.mean(chosen_rewards)),
            "rejected_rewards": float(np.mean(rejected_rewards)),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
        }
