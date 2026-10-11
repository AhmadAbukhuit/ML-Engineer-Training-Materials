# Phase 8: Reinforcement Learning

Welcome to the **Reinforcement Learning** phase of the Machine Learning Engineering curriculum. This phase covers decision-making under uncertainty, Markov Decision Processes, dynamic programming, tabular Q-learning, deep value-based methods (DQN, DDQN, Dueling), policy gradients (REINFORCE, PPO, SAC), and LLM alignment (RLHF, DPO).

---

## 📚 Curriculum & Reference Guides

- [`01_reinforcement_learning_models_catalog.md`](file:///home/ahmad/ML-Engineer-Training-Materials/08_reinforcement_learning/01_reinforcement_learning_models_catalog.md): Comprehensive mathematical catalog of all reinforcement learning algorithms (MDPs, Bellman Equations, Policy Iteration, Value Iteration, Monte Carlo, SARSA, Q-Learning, DQN, Double DQN, Dueling DQN, PER, Rainbow, REINFORCE, Actor-Critic A2C, TRPO, PPO, DDPG, TD3, SAC, World Models, DreamerV3, RLHF, and DPO).

---

## 🛠️ Code Implementations (`src/`)

- [`src/gridworld.py`](file:///home/ahmad/ML-Engineer-Training-Materials/08_reinforcement_learning/src/gridworld.py): Self-contained 2D GridWorld MDP environment with obstacles, step penalties, and goal rewards.
- [`src/q_learning.py`](file:///home/ahmad/ML-Engineer-Training-Materials/08_reinforcement_learning/src/q_learning.py): Tabular Q-Learning agent with $\epsilon$-greedy exploration decay and off-policy Bellman optimality updates.
- [`src/dqn_agent.py`](file:///home/ahmad/ML-Engineer-Training-Materials/08_reinforcement_learning/src/dqn_agent.py): Deep Q-Network agent in NumPy featuring cyclic Experience Replay Buffer and periodic Target Network weight synchronization.
- [`src/policy_gradient.py`](file:///home/ahmad/ML-Engineer-Training-Materials/08_reinforcement_learning/src/policy_gradient.py): REINFORCE Monte Carlo policy gradient agent with softmax action distribution and baseline subtraction variance reduction.

---

## 🧪 Unit Tests (`tests/`)

Run the reinforcement learning test suite using pytest:

```bash
pytest 08_reinforcement_learning/tests
```
