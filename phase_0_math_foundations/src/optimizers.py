"""
This code implements foundational gradient descent-based optimization algorithms.
Optimizers adjust the neural network's weights based on the computed gradients to minimize the loss.
We implement four variants: standard SGD, SGD with Momentum, RMSprop, and Adam, to demonstrate 
the evolution of optimization techniques from simple gradient steps to adaptive learning rates 
and momentum-based updates.
"""
class Optimizer:
    """
    Abstract base class for all optimizers.
    """
    def __init__(self, parameters, lr=0.01):
        """
        Initializes the optimizer with a list of model parameters and a learning rate.
        """
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """
        Resets the gradients of all parameters to zero. 
        Must be called before the backward pass to prevent gradient accumulation.
        """
        for p in self.parameters:
            p.grad = 0.0

    def step(self):
        """Updates the parameters using the calculated gradients."""
        raise NotImplementedError("Step method must be implemented by subclasses.")


class SGD(Optimizer):
    """
    Standard Stochastic Gradient Descent (SGD).
    Mathematical Update: theta = theta - lr * grad
    
    Description: It takes a step in the exact opposite direction of the gradient. 
    While simple, it can be slow to navigate ravines in the loss landscape and 
    is highly sensitive to the choice of learning rate.
    """
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad


class SGDMomentum(Optimizer):
    """
    SGD with Momentum.
    Mathematical Update: 
      v = beta * v + lr * grad
      theta = theta - v
      
    Description: Introduces a 'velocity' term (v) that acts like physical momentum. 
    It dampens oscillations across ravines and accelerates convergence by accumulating 
    gradients from past steps. The 'beta' parameter controls friction (typically 0.9).
    """
    def __init__(self, parameters, lr=0.01, beta=0.9):
        super().__init__(parameters, lr)
        self.beta = beta
        # Initialize velocity for each parameter to 0
        self.velocities = [0.0 for _ in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            self.velocities[i] = self.beta * self.velocities[i] + self.lr * p.grad
            p.data -= self.velocities[i]


class RMSprop(Optimizer):
    """
    RMSprop (Root Mean Square Propagation).
    
    Description: Adapts the learning rate individually for each parameter. 
    It maintains a moving average of the squared gradients (s) and scales the 
    learning rate inversely proportional to the square root of this average. 
    This allows larger steps for parameters with small gradients and smaller 
    steps for parameters with large gradients, preventing exploding gradients.
    """
    def __init__(self, parameters, lr=0.01, beta=0.99, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta = beta
        self.eps = eps
        # Initialize squared gradient moving average (s) to 0
        self.s = [0.0 for _ in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            # 1. Update the moving average of squared gradients
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (p.grad ** 2)
            
            # 2. Parameter update: divide by sqrt(s) + epsilon (for numerical stability)
            p.data -= (self.lr / ((self.s[i] ** 0.5) + self.eps)) * p.grad


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation).
    
    Description: The default standard for deep learning. It essentially combines 
    the benefits of Momentum (first moment, m) and RMSprop (second moment, v). 
    It also includes a bias correction mechanism (m_hat, v_hat) to fix the issue 
    where the initial moments are biased toward zero at the start of training.
    """
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # First moment (momentum-like velocity)
        self.m = [0.0 for _ in self.parameters]
        # Second raw moment (RMSprop-like squared gradients)
        self.v = [0.0 for _ in self.parameters]
        # Time step counter for bias correction
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            # 1. Update biased first moment estimate (mean of gradients)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            
            # 2. Update biased second raw moment estimate (uncentered variance of gradients)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            # 3. Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # 4. Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 5. Final parameter update
            p.data -= (self.lr / ((v_hat ** 0.5) + self.eps)) * m_hat