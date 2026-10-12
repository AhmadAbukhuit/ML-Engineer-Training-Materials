# Convolutional Networks & Deep Residual Architectures

Convolutional Neural Networks (CNNs) and Residual Networks (ResNets) revolutionized visual perception by enforcing **spatial equivariance** and solving the **vanishing gradient problem** in deep neural graphs.

---

## 1. Principles of 2D Convolution

Unlike fully-connected layers where every input unit connects to every output unit, convolutional layers rely on three structural properties:

1. **Local Receptive Fields:** Each kernel connects only to a localized spatial patch of the input.
2. **Weight Sharing:** The same kernel weights are swept across the entire input grid, drastically reducing the total number of free parameters.
3. **Translation Equivariance:** If an object shifts in the input image, its feature map representation shifts by the exact same amount:
   $$f(g(x)) = g(f(x))$$

### The 2D Convolution Formula

For an input image $I$ and kernel $K$ of height $2m+1$ and width $2n+1$:

$$S(i, j) = (I * K)(i, j) = \sum_{u=-m}^m \sum_{v=-n}^n I(i - u, j - v) K(u, v)$$

### Output Spatial Dimension Calculation

Given input dimension $W_{\text{in}}$, kernel size $K$, padding $P$, and stride $S$:

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K + 2P}{S} \right\rfloor + 1$$

---

## 2. Receptive Field & Downsampling

* **Receptive Field:** The spatial extent in the input image that directly influences the activation of a neuron in layer $L$.
* Stacking two $3 \times 3$ convolutions yields a receptive field of $5 \times 5$, while requiring fewer parameters $(2 \times 3^2 = 18)$ than a single $5 \times 5$ kernel $(25)$ and introducing an extra non-linear activation.
* **Pooling / Strided Convolutions:** Reduce spatial dimensions while increasing channel depth, allowing deeper layers to capture high-level semantic concepts (e.g., ears, faces) rather than low-level edges.

---

## 3. The Deep Degradation Problem & ResNets

Before ResNets, adding more layers to a deep neural network caused accuracy to saturate and then degrade rapidly—not due to overfitting (since training error also increased), but due to optimization difficulty and vanishing gradients.

### The Residual Learning Solution

Instead of forcing layers to directly fit an underlying mapping $\mathcal{H}(\mathbf{x})$, ResNet parameterizes the layers to fit a residual mapping $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

### Mathematical Advantage During Backpropagation

Consider the gradient with respect to input $\mathbf{x}$ across $L$ residual blocks:

$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \left( I + \sum_{l=1}^{L-1} \frac{\partial \mathcal{F}_l}{\partial \mathbf{x}_l} \right)$$

* The additive identity term $I$ creates a clean "gradient superhighway".
* Even if the layer weights $\mathcal{F}_l$ produce small gradients, the error signal can bypass them and propagate directly from the final loss to the first layer without diminishing.
