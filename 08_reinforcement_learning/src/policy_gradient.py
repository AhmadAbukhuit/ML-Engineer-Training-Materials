"""REINFORCE Policy Gradient Agent from First Principles in NumPy."""

import numpy as np


class REINFORCEAgent:
    """Monte Carlo Policy Gradient (REINFORCE) Agent.

    Parameters
    ----------
    state_dim : int
        Dimension of state features.
    action_dim : int
        Number of discrete actions.
    lr : float
        Policy gradient learning rate.
    gamma : float
        Discount factor.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.01,
        gamma: float = 0.99,
        random_state: int | None = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.rng = np.random.default_rng(random_state)

        # Policy weights: shape (action_dim, state_dim)
        self.theta = self.rng.normal(0, 0.1, (action_dim, state_dim))

        # Trajectory storage
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute softmax policy distribution pi(a | s)."""
        state = np.asarray(state, dtype=np.float64)
        logits = np.dot(self.theta, state)
        # Numerical stability shift
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy distribution pi(a | s)."""
        probs = self.get_action_probs(state)
        return int(self.rng.choice(self.action_dim, p=probs))

    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        """Record step in the current trajectory."""
        self.states.append(np.asarray(state, dtype=np.float64))
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_returns(self) -> np.ndarray:
        """Compute discounted cumulative returns G_t."""
        T = len(self.rewards)
        returns = np.zeros(T, dtype=np.float64)
        running_g = 0.0
        for t in reversed(range(T)):
            running_g = self.rewards[t] + self.gamma * running_g
            returns[t] = running_g
        return returns

    def finish_episode(self) -> float:
        """Perform Monte Carlo policy gradient update at the end of an episode."""
        T = len(self.rewards)
        if T == 0:
            return 0.0

        returns = self.compute_returns()
        # Baseline: subtract mean return to reduce gradient variance
        baseline = float(np.mean(returns))
        advantages = returns - baseline

        # Accumulate policy gradient: d/d theta log pi(a_t | s_t) * G_t
        grad = np.zeros_like(self.theta)

        for t in range(T):
            s_t = self.states[t]
            a_t = self.actions[t]
            adv_t = advantages[t]

            probs = self.get_action_probs(s_t)

            # d/d theta_a log pi(a | s) = s * (I(a == a_t) - pi(a | s))
            for a in range(self.action_dim):
                indicator = 1.0 if a == a_t else 0.0
                grad[a] += adv_t * (indicator - probs[a]) * s_t

        # Gradient ascent update with clipping
        clip_val = 1.0
        grad = np.clip(grad, -clip_val, clip_val)
        self.theta += self.lr * grad

        total_reward = float(np.sum(self.rewards))
        # Clear trajectory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        return total_reward
