import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gridworld import GridWorld
from q_learning import QLearningAgent


def test_gridworld_mechanics():
    env = GridWorld(height=3, width=3, start_pos=(0, 0), goal_pos=(2, 2))
    assert env.reset() == 0

    # Move right (action 1) -> (0, 1), state = 1
    next_s, r, done, _ = env.step(1)
    assert next_s == 1
    assert r == env.step_penalty
    assert not done


def test_q_learning_train_and_solve_gridworld():
    env = GridWorld(height=3, width=3, start_pos=(0, 0), goal_pos=(2, 2), step_penalty=-0.05)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.2,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.98,
        random_state=42,
    )

    rewards = agent.train(env, episodes=200, max_steps=50)
    assert len(rewards) == 200

    # Policy evaluation should achieve positive return (reaching goal)
    eval_return = agent.evaluate(env, episodes=5, max_steps=20)
    assert eval_return > 0.0
