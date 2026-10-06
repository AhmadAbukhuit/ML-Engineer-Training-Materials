# Optimization Algorithms

Having defined a loss function and computed gradients via backpropagation, we must update the network's weights. The goal is to navigate the high-dimensional loss surface to find the global minimum quickly and reliably.

## The Loss Surface & Learning Rate

Imagine dropping a ball into a mountainous landscape. The ball wants to roll to the lowest valley.

* **Convex Surface:** A perfect bowl shape. Classical ML algorithms (Linear Regression, SVMs) have convex surfaces, meaning they guarantee a single, global minimum.
* **Non-Convex Surface:** Deep Neural Networks create extremely jagged surfaces with many local valleys (local minima) and flat plains (plateaus).
* **The Learning Rate ($\eta$):** Defines the step size. If $\eta$ is too small, training takes weeks and gets stuck in local minima. If $\eta$ is too large, the optimizer overshoots the valley, causing the loss to explode to `NaN`.

## Data Batching Strategies

Calculating the true gradient requires evaluating the entire dataset, which is impossible if you have 100GB of images.

* **Batch Gradient Descent:** Calculates gradient using the whole dataset. Slow, but accurate.
* **Stochastic Gradient Descent (SGD):** Calculates gradient using exactly one sample. Lightning fast, but highly erratic and noisy.
* **Mini-Batch Gradient Descent:** The industry standard. Calculates gradient using a small chunk (e.g., 32, 64, or 128 samples). Balances speed, vectorization efficiency, and gradient stability.

## Standard SGD & Momentum

### Standard SGD Update Equation

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

* **The Flaw:** It struggles in "ravines" (areas where the surface curves steeply in one dimension but is flat in another), causing it to oscillate wildly across the walls rather than moving down the center.

### SGD with Momentum

Simulates the physics of a ball rolling down a hill, accumulating velocity.

$$v_{t+1} = \beta v_t + \eta \nabla L(\theta_t)$$

$$\theta_{t+1} = \theta_t - v_{t+1}$$

* **Intuition:** $\beta$ (usually $0.9$) dictates how much previous velocity to keep. If the gradient keeps pointing in the same direction, velocity grows, accelerating the optimizer. If the gradient rapidly changes directions (oscillating across a ravine), those opposing forces cancel out, dampening the jitter.

## RMSprop (Root Mean Square Propagation)

RMSprop addresses the ravine problem differently. It assigns a custom, adaptive learning rate to every single parameter.

### The Equation

$$s_{t+1} = \beta s_t + (1 - \beta) (\nabla L(\theta_t))^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1}} + \epsilon} \nabla L(\theta_t)$$

* **Intuition:** It keeps a running average of the squared gradients ($s_t$). During the update, it divides the base learning rate by the square root of this average.
* **Result:** If a parameter has huge, oscillating gradients, dividing by a large $s_t$ shrinks its effective learning rate, stabilizing it. If a parameter has tiny, flat gradients, dividing by a small $s_t$ boosts its learning rate.

## Adam (Adaptive Moment Estimation)

Adam is the default standard for deep learning. It combines the best features of Momentum (first moment: velocity) and RMSprop (second moment: adaptive scaling).

### The Update Equation

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(Momentum)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(RMSprop)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(Bias Correction)}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t \quad \text{(Update)}$$

* **Intuition:** Because $m_t$ and $v_t$ are initialized to $0$, they are heavily biased toward $0$ in the early training steps. The "hat" variables ($\hat{m}_t, \hat{v}_t$) apply a mathematical bias correction to fix this cold-start problem.
* **Primary Advantage:** Incredibly robust out-of-the-box. It converges rapidly on complex architectures like Vision Transformers and deep CNNs without requiring exhaustive manual learning rate tuning.
