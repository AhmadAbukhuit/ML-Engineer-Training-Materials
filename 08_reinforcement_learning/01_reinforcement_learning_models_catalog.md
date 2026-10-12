# Complete Catalog of Reinforcement Learning Models

Reinforcement Learning (RL) is the computational paradigm wherein an active agent learns an optimal behavioral policy $\pi^*(a|s)$ through trial-and-error sequential interactions with an environment, maximizing expected cumulative discounted return:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t, s_{t+1}) \right]$$

---

## 1. Mathematical Foundations & Markov Decision Processes (MDP)

An infinite-horizon discounted Markov Decision Process is formally defined as the 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

- $\mathcal{S}$: State space.
- $\mathcal{A}$: Action space (discrete or continuous).
- $\mathcal{P}(s' | s, a) = \mathbb{P}(S_{t+1} = s' | S_t = s, A_t = a)$: Transition probability function satisfying the **Markov Property**:
  $$\mathbb{P}(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = \mathbb{P}(S_{t+1} | S_t, A_t)$$
- $\mathcal{R}(s, a, s')$: Reward function mapping transitions to immediate scalar rewards.
- $\gamma \in [0, 1)$: Discount factor ensuring geometric convergence of infinite returns:
  $$G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

### 1.1 Value Functions & Bellman Equations

- **State-Value Function $V^\pi(s)$**: Expected return starting from state $s$ following policy $\pi$:
  $$V^\pi(s) = \mathbb{E}_\pi [G_t | S_t = s] = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s', r} \mathcal{P}(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]$$
- **Action-Value Function $Q^\pi(s, a)$**: Expected return starting from state $s$, taking action $a$, and thereafter following $\pi$:
  $$Q^\pi(s, a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a] = \sum_{s', r} \mathcal{P}(s', r | s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]$$
- **Bellman Optimality Equations**:
  $$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s', r} \mathcal{P}(s', r | s, a) \left[ r + \gamma V^*(s') \right]$$
  $$Q^*(s, a) = \sum_{s', r} \mathcal{P}(s', r | s, a) \left[ r + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]$$

---

## 2. Dynamic Programming & Model-Based Methods

Assumes full knowledge of environmental transition dynamics $\mathcal{P}$ and rewards $\mathcal{R}$.

### 2.1 Policy Evaluation

- Computes $V^\pi$ for a fixed policy $\pi$ via iterative application of the Bellman expectation backup operator $T^\pi$:
  $$V_{k+1}(s) \leftarrow \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')]$$
  Guaranteed convergence by Banach Fixed Point Theorem (contraction mapping with factor $\gamma$).

### 2.2 Policy Iteration (Generalized Policy Iteration / GPI)

- Alternates between complete Policy Evaluation ($V^{\pi_k}$) and Policy Improvement:
  $$\pi_{k+1}(s) = \arg\max_{a \in \mathcal{A}} \sum_{s', r} p(s', r | s, a) [r + \gamma V^{\pi_k}(s')]$$
  Terminates at strictly optimal policy $\pi^*$ in finite iterations.

### 2.3 Value Iteration

- Collapses policy evaluation and greedy improvement into a single step via the Bellman optimality operator $T^*$:
  $$V_{k+1}(s) \leftarrow \max_{a \in \mathcal{A}} \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')]$$

---

## 3. Model-Free Tabular RL

Learns directly from environmental experience without knowing $\mathcal{P}$ or $\mathcal{R}$.

### 3.1 Monte Carlo Methods

- Updates value estimates using complete empirical episode returns:
  $$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$
- **First-Visit vs Every-Visit**: Considers only the first occurrence of state $s$ in an episode or every occurrence.
- **Unbiased** ($ \mathbb{E}[G_t] = V^\pi(s)$) but exhibits high variance; strictly requires episodic environments.

### 3.2 Temporal Difference (TD) Learning

- Bootstraps estimates using immediate reward plus current discounted value estimate of next state.
- **TD(0) Prediction**:
  $$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$
- **SARSA (On-Policy TD Control)**:
  - Generates experience $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ where $A_{t+1} \sim \pi(\cdot|S_{t+1})$:
    $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
- **Q-Learning (Off-Policy TD Control, Watkins 1989)**:
  - Updates target independently of policy action selection via $\max_{a'} Q(S_{t+1}, a')$:
    $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$
- **Expected SARSA**:
  - Replaces max with expected value: $\mathbb{E}_{a \sim \pi}[Q(S_{t+1}, a)]$, reducing update variance.
- **n-Step TD & $\text{TD}(\lambda)$**:
  - Bridges TD(0) and Monte Carlo via eligibility traces $E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbb{I}(S_t = s)$.

---

## 4. Value-Based Deep Reinforcement Learning

Replaces lookup tables with deep parameterized function approximators $Q(s, a; \boldsymbol{\theta})$.

### 4.1 The Deadly Triad & Deep Q-Networks (DQN)

- **The Deadly Triad**: Instability/divergence occurs when combining:
  1. *Function approximation* (neural networks).
  2. *Bootstrapping* (using current estimates to update targets).
  3. *Off-policy learning* (learning about optimal policy while behaving via exploratory policy).
- **DQN Innovations (Mnih et al., 2015)**:
  - **Experience Replay Buffer $\mathcal{D}$**: Stores transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ and samples mini-batches uniformly at random, breaking temporal autocorrelation and ensuring i.i.d. training.
  - **Separate Target Network $\boldsymbol{\theta}^-$**: Stabilizes non-stationary targets:
    $$y_i = r_i + \gamma (1 - d_i) \max_{a'} Q(s'_i, a'; \boldsymbol{\theta}^-)$$
    $$\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( y_i - Q(s, a; \boldsymbol{\theta}) \right)^2 \right]$$
    Target network weights are frozen and synchronized every $C$ steps or via Polyak smoothing: $\boldsymbol{\theta}^- \leftarrow \tau \boldsymbol{\theta} + (1 - \tau) \boldsymbol{\theta}^-$.

### 4.2 Double DQN (DDQN)

- **Maximization Bias**: Standard DQN suffers from severe overestimation bias because $\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$.
- **Decoupled Update**: Uses online network $\boldsymbol{\theta}$ to **select** the greedy action, and target network $\boldsymbol{\theta}^-$ to **evaluate** its value:
  $$y_i^{\text{DDQN}} = r_i + \gamma (1 - d_i) Q\left( s'_i, \arg\max_{a'} Q(s'_i, a'; \boldsymbol{\theta}); \boldsymbol{\theta}^- \right)$$

### 4.3 Dueling DQN

- Decomposes $Q(s, a)$ into a state value function $V(s)$ and state-dependent action advantage function $A(s, a)$:
  $$Q(s, a; \boldsymbol{\theta}, \boldsymbol{\alpha}, \boldsymbol{\beta}) = V(s; \boldsymbol{\theta}, \boldsymbol{\beta}) + \left( A(s, a; \boldsymbol{\theta}, \boldsymbol{\alpha}) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \boldsymbol{\theta}, \boldsymbol{\alpha}) \right)$$
- The mean subtraction ensures unique identifiability: at the optimal action, $A^*(s, a^*) = 0$ and $Q^*(s, a^*) = V^*(s)$.

### 4.4 Prioritized Experience Replay (PER)

- Samples transitions with probability proportional to absolute TD error $|\delta_i|^\alpha$:
  $$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon$$
- Corrects sampling bias using Importance Sampling (IS) weights: $w_i = (N \cdot P(i))^{-\beta}$.

### 4.5 Rainbow DQN

- Fuses 6 DQN advancements into a single architecture: Double Q-learning, Dueling Networks, Prioritized Replay, Multi-step learning ($n$-step returns), Distributional RL (Categorical C51), and Noisy Networks (parametric exploration noise).

---

## 5. Policy Gradient & Actor-Critic Architectures

Directly parameterizes the policy $\pi_\theta(a|s)$ and optimizes performance objective $J(\theta)$ via gradient ascent.

### 5.1 The Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) Q^{\pi_\theta}(s_t, a_t) \right]$$
Remarkably, the gradient does *not* require computing derivatives of environmental state transition probabilities $\nabla_\theta \mathcal{P}(s'|s, a)$.

### 5.2 REINFORCE (Monte Carlo Policy Gradient)

- Uses empirical sample return $G_t$ as an unbiased estimator of $Q(s_t, a_t)$:
  $$\theta \leftarrow \theta + \alpha \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \left( G_t - b(s_t) \right)$$
- Baseline $b(s_t)$ (e.g., learned state value $V(s_t)$) reduces variance without introducing bias.

### 5.3 Advantage Actor-Critic (A2C / A3C)

- **Actor**: Parameterizes policy $\pi_\theta(a|s)$.
- **Critic**: Parameterizes value function $V_\phi(s)$.
- Replaces $G_t$ with the Advantage function $A(s, a) = Q(s, a) - V(s) \approx r + \gamma V_\phi(s') - V_\phi(s)$ (TD error).
- **A3C**: Asynchronous parallel worker threads update a central global network.
- **A2C**: Synchronous, deterministic batched GPU execution replacing async locks.

### 5.4 Trust Region Policy Optimization (TRPO)

- Solves policy degradation caused by large gradient steps. Maximizes surrogate objective subject to a KL-divergence constraint on policy shift:
  $$\max_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right] \quad \text{s.t.} \quad \hat{\mathbb{E}}_t [D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s_t) \parallel \pi_\theta(\cdot|s_t))] \le \delta$$
- Solved via Second-order Natural Policy Gradient (Fisher Information Matrix and Conjugate Gradient).

### 5.5 Proximal Policy Optimization (PPO)

- Achieves TRPO's monotonic improvement guarantee using a first-order clipped surrogate objective:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
  where probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ and $\epsilon \approx 0.2$.
- Industry gold-standard for robotics, game AI, and LLM Alignment via Reinforcement Learning from Human Feedback (RLHF).

---

## 6. Continuous Control & Maximum Entropy RL

### 6.1 DDPG & TD3

- **DDPG (Deep Deterministic Policy Gradient)**:
  - Deterministic policy $\mu_\theta(s) \in \mathbb{R}^d$.
  - Continuous action space; uses replay buffer and target networks.
- **TD3 (Twin Delayed DDPG)**:
  - Clipped Double Q-learning (two critic networks $Q_{\phi_1}, Q_{\phi_2}$; target takes minimum).
  - Delayed policy updates (updates actor less frequently than critics).
  - Target policy smoothing (adds clipped noise to target actions).

### 6.2 SAC (Soft Actor-Critic)

- Optimizes **Maximum Entropy RL**: agent maximizes both expected reward and policy entropy:
  $$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t)} [R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$
  where temperature $\alpha$ controls exploration vs exploitation.
- Off-policy, sample-efficient, highly stable across complex continuous control benchmarks.

---

## 7. Modern Frontiers: Model-Based DRL & LLM Alignment

### 7.1 Model-Based DRL (DreamerV3, MuZero)

- **DreamerV3 (Hafner et al.)**: Learns a Recurrent State-Space World Model (RSSM) predicting latent transition dynamics, then trains an Actor-Critic completely in imagined world-model rollouts.
- **MuZero (Silver et al.)**: Combines tree search (MCTS) with learned latent models without environmental rules.

### 7.2 RL from Human Feedback (RLHF) & Direct Preference Optimization (DPO)

- **RLHF**:
  1. Train Reward Model $r_\psi(x, y)$ on pairwise human preferences using Bradley-Terry loss:
     $$\mathcal{L}_R = -\log \sigma(r_\psi(x, y_w) - r_\psi(x, y_l))$$
  2. Optimize LLM policy via PPO against reward model with KL penalty to prevent reward hacking:
     $$\max_\theta \mathbb{E}_{(x, y) \sim \pi_\theta} [r_\psi(x, y)] - \beta D_{\text{KL}}(\pi_\theta(y|x) \parallel \pi_{\text{ref}}(y|x))$$
- **DPO (Rafailov et al.)**: Replaces PPO RL training with an exact closed-form objective directly optimizing the policy on preference pairs without explicit reward modeling:
  $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

## 8. Summary Comparison Matrix

| Algorithm | Paradigm | Action Space | On/Off Policy | Sample Efficiency | Key Use Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q-Learning** | Tabular Value-based | Discrete | Off-Policy | High (in small state spaces) | Gridworlds, finite MDPs |
| **DQN / DDQN** | Deep Value-based | Discrete | Off-Policy | Moderate | Atari, game playing, discrete control |
| **REINFORCE** | Policy Gradient | Discrete / Continuous | On-Policy | Low (high variance) | Foundational baseline |
| **A2C / A3C** | Actor-Critic | Discrete / Continuous | On-Policy | Moderate | Parallel simulated environments |
| **PPO** | Clipped Policy Gradient | Discrete / Continuous | On-Policy | High (among on-policy) | LLM Alignment (RLHF), Robotics |
| **SAC** | Maximum Entropy Actor-Critic | Continuous | Off-Policy | Very High | Continuous robotics, autonomous driving |
| **DPO** | Direct Preference Optimization | Token sequences | Off-Policy | Very High | LLM Alignment from preference data |
