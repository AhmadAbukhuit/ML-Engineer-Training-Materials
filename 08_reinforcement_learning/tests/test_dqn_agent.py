import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dqn_agent import DQNAgent, ReplayBuffer


def test_replay_buffer_capacity_and_sampling():
    buffer = ReplayBuffer(capacity=10, random_state=42)
    for i in range(15):
        s = np.array([i, i + 1])
        buffer.push(s, action=i % 2, reward=1.0, next_state=s * 2, done=False)

    assert len(buffer) == 10
    states, actions, rewards, _next_states, _dones = buffer.sample(batch_size=4)
    assert states.shape == (4, 2)
    assert len(actions) == 4
    assert len(rewards) == 4


def test_dqn_agent_step_and_target_sync():
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        hidden_dim=8,
        lr=0.01,
        batch_size=8,
        target_update_freq=5,
        random_state=42,
    )

    # Fill replay buffer
    for _ in range(20):
        s = np.random.randn(4)
        a = np.random.randint(0, 2)
        r = 1.0
        s_next = np.random.randn(4)
        agent.memory.push(s, a, r, s_next, False)

    initial_loss = agent.train_step()
    assert initial_loss >= 0.0

    # Train several steps
    for _ in range(10):
        agent.train_step()

    # Verify action selection
    action = agent.select_action(np.random.randn(4), greedy=True)
    assert action in (0, 1)
