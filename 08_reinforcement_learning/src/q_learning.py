"""Tabular Q-Learning algorithm from First Principles."""

from typing import Any
import numpy as np


class QLearningAgent:
    """Tabular Q-Learning Agent.

    Parameters
    ----------
    n_states : int
        Number of discrete states.
    n_actions : int
        Number of discrete actions.
    alpha : float
        Learning rate (0, 1].
    gamma : float
        Discount factor [0, 1].
    epsilon : float
        Initial exploration rate [0, 1].
    epsilon_min : float
        Minimum exploration rate floor.
    epsilon_decay : float
        Multiplicative decay factor per episode.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        random_state: int | None = 42,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(random_state)

        # Initialize Q-table to zeros
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Select action via epsilon-greedy policy."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        # Break ties randomly among best actions
        q_vals = self.q_table[state]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Off-policy Bellman optimality update.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') * (1 - done) - Q(s, a)]
        """
        current_q = self.q_table[state, action]
        next_max_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * next_max_q
        td_error = td_target - current_q

        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: Any, episodes: int = 500, max_steps: int = 100) -> list[float]:
        """Train agent on environment."""
        episode_rewards: list[float] = []

        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            self.decay_epsilon()
            episode_rewards.append(total_reward)

        return episode_rewards

    def evaluate(self, env: Any, episodes: int = 10, max_steps: int = 100) -> float:
        """Evaluate agent deterministically (greedy policy)."""
        eval_rewards = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                action = self.select_action(state, greedy=True)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                if done:
                    break
            eval_rewards.append(total_reward)
        return float(np.mean(eval_rewards))
