# Multivariate Calculus & Computational Graphs

Calculus tells us how a function changes when we tweak its inputs. In machine learning, we need to know how the "loss" changes when we tweak millions of "weights." We use this information to update the weights and minimize the loss.

## The Derivative (Rate of Change)

Before jumping to multiple variables, we must understand the basic derivative. A derivative measures the instantaneous rate of change of a function.

**The Definition:**

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

* **Intuition:** If I nudge my input $x$ by a microscopic amount $h$, how much does the output $f(x)$ change? If $f'(x) = 5$, it means the output grows 5 times faster than the input.
* **ML Application:** If $x$ is a weight and $f(x)$ is the loss, a positive derivative means increasing the weight increases the loss (bad). A negative derivative means increasing the weight decreases the loss (good).

## Partial Derivatives

Neural networks do not have one weight; they have millions. A partial derivative measures how a function changes when we nudge one specific variable, treating all other variables as frozen constants.

**Notation:**

$$\frac{\partial f(x, y)}{\partial x}$$

* **Example:** For $f(x, y) = x^2 y + y^3$:
    * $\frac{\partial f}{\partial x} = 2xy$ (Treating $y$ as a constant number).
    * $\frac{\partial f}{\partial y} = x^2 + 3y^2$ (Treating $x$ as a constant number).
* **ML Application:** We use partial derivatives to isolate the exact contribution of a single weight (like $w_{12}$) to the final error of the network, ignoring all other millions of weights for that split second.

## The Gradient Vector

The gradient is simply a vector that groups together all the partial derivatives of a function.

**The Formula:**

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

* **Intuition:** If you are standing on a hilly, multi-dimensional landscape (the loss surface), the gradient vector always points in the direction of the steepest uphill slope.
* **ML Application:** To train a model, we want to find the bottom of the valley (minimum error). Therefore, Gradient Descent takes a step in the opposite direction of the gradient ($-\nabla f$).

## Multivariate Chain Rule & Computational Graphs

The chain rule allows us to calculate the derivative of complex, nested functions by multiplying the derivatives of their inner parts.

**The Formula:**

Given $z = f(y)$ and $y = g(x)$:

$$\frac{\partial z}{\partial x_i} = \sum_{j} \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

* **Intuition:** If variable $A$ affects $B$ twice as fast, and $B$ affects $C$ three times as fast, then $A$ affects $C$ six times as fast ($2 \times 3$). If there are multiple paths from $A$ to $C$, you sum them up.
* **ML Application:** This is the mathematical engine behind Backpropagation. Neural networks are computational graphs (Layer 3 wraps Layer 2, which wraps Layer 1). The chain rule lets us traverse backwards from the loss, multiplying local derivatives to calculate exactly how a weight in Layer 1 impacts the final loss.

## Jacobian & Hessian Matrices

While the gradient is used for functions that output a single number (a scalar loss), the Jacobian is used for functions that output a vector. The Hessian takes it a step further to second-order derivatives.

**The Jacobian:**

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

* **Use Case:** Calculating the derivative of a Softmax layer (which maps a vector of logits to a vector of probabilities).

**The Hessian**

A square matrix of second-order partial derivatives ($\frac{\partial^2 f}{\partial x_i \partial x_j}$).

* **Use Case:** It describes the curvature of the loss landscape (is it shaped like a bowl or a saddle?). Used in advanced second-order optimization techniques (like Newton's Method) which converge in fewer steps but are too computationally expensive for deep learning.
