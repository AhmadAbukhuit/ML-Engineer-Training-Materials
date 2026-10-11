import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from policy_gradient import REINFORCEAgent


def test_reinforce_action_sampling_and_update():
    agent = REINFORCEAgent(state_dim=3, action_dim=2, lr=0.05, gamma=0.99, random_state=42)

    s = np.array([1.0, 0.5, -0.5])
    probs = agent.get_action_probs(s)
    assert len(probs) == 2
    np.testing.assert_allclose(np.sum(probs), 1.0)

    # Simulate an episode
    for step in range(5):
        action = agent.select_action(s)
        reward = 1.0 if action == 1 else -1.0
        agent.store_transition(s, action, reward)

    total_return = agent.finish_episode()
    assert isinstance(total_return, float)
    # Buffers should be cleared after episode finish
    assert len(agent.states) == 0
    assert len(agent.actions) == 0
