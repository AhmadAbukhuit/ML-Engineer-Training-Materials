"""Deep Q-Network (DQN) Agent from First Principles in NumPy."""

import numpy as np


class ReplayBuffer:
    """Cyclic Experience Replay Buffer for DQN."""

    def __init__(self, capacity: int = 10000, random_state: int | None = 42) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(random_state)
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """Save a transition tuple."""
        transition = (
            np.asarray(state, dtype=np.float64),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float64),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions uniformly at random."""
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            np.array(states, dtype=np.float64),
            np.array(actions, dtype=int),
            np.array(rewards, dtype=np.float64),
            np.array(next_states, dtype=np.float64),
            np.array(dones, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class MLPNetwork:
    """2-Layer Multi-Layer Perceptron Q-Function Approximator in NumPy."""

    def __init__(
        self, state_dim: int, hidden_dim: int, action_dim: int, rng: np.random.Generator
    ) -> None:
        self.W1 = rng.normal(0, np.sqrt(2.0 / state_dim), (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, action_dim))
        self.b2 = np.zeros(action_dim)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Forward pass with activation caching."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        a1 = np.dot(x, self.W1) + self.b1
        h1 = np.maximum(0.0, a1)
        q_values = np.dot(h1, self.W2) + self.b2
        cache = {"x": x, "a1": a1, "h1": h1}
        return q_values, cache

    def copy_weights_from(self, source: "MLPNetwork") -> None:
        """Hard update copying weights from another network."""
        self.W1 = source.W1.copy()
        self.b1 = source.b1.copy()
        self.W2 = source.W2.copy()
        self.b2 = source.b2.copy()


class DQNAgent:
    """Deep Q-Network (DQN) Agent.

    Features:
    - Experience Replay
    - Separate Target Network
    - Epsilon-greedy action selection
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        lr: float = 0.005,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.99,
        target_update_freq: int = 20,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        random_state: int | None = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        self.rng = np.random.default_rng(random_state)
        self.q_net = MLPNetwork(state_dim, hidden_dim, action_dim, self.rng)
        self.target_net = MLPNetwork(state_dim, hidden_dim, action_dim, self.rng)
        self.target_net.copy_weights_from(self.q_net)

        self.memory = ReplayBuffer(capacity=buffer_capacity, random_state=random_state)
        self.step_counter = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action via epsilon-greedy."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_dim))

        q_vals, _ = self.q_net.forward(state)
        return int(np.argmax(q_vals[0]))

    def train_step(self) -> float:
        """Sample mini-batch and perform one gradient descent update."""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Compute Q(s, :) for online network
        q_preds, cache = self.q_net.forward(states)

        # Compute Q_target(s', :) for target network
        next_q_targets, _ = self.target_net.forward(next_states)
        max_next_q = np.max(next_q_targets, axis=1)

        # Bellman target: y = r + gamma * max_a' Q_target(s', a') * (1 - done)
        targets = q_preds.copy()
        for i in range(self.batch_size):
            target_val = rewards[i] if dones[i] else rewards[i] + self.gamma * max_next_q[i]
            targets[i, actions[i]] = target_val

        # Loss & analytical backprop
        td_errors = q_preds - targets  # only non-zero at actions[i]
        loss = float(np.mean(td_errors**2))

        # Gradient wrt q_preds: 2 * td_errors / batch_size
        dQ = 2.0 * td_errors / self.batch_size

        # Layer 2 gradients
        dW2 = np.dot(cache["h1"].T, dQ)
        db2 = np.sum(dQ, axis=0)
        dh1 = np.dot(dQ, self.q_net.W2.T)

        # Layer 1 gradients
        da1 = dh1 * (cache["a1"] > 0).astype(np.float64)
        dW1 = np.dot(cache["x"].T, da1)
        db1 = np.sum(da1, axis=0)

        # Gradient clipping
        clip_val = 1.0
        dW2 = np.clip(dW2, -clip_val, clip_val)
        db2 = np.clip(db2, -clip_val, clip_val)
        dW1 = np.clip(dW1, -clip_val, clip_val)
        db1 = np.clip(db1, -clip_val, clip_val)

        # Update online network weights
        self.q_net.W2 -= self.lr * dW2
        self.q_net.b2 -= self.lr * db2
        self.q_net.W1 -= self.lr * dW1
        self.q_net.b1 -= self.lr * db1

        # Periodic target sync
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.q_net)

        return loss

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
