# Linear Algebra & Matrix Decompositions

Linear algebra is the language of machine learning. Neural networks process data in batches, meaning single variables (scalars) are scaled up to arrays (vectors), grids (matrices), and higher-dimensional structures (tensors).

## What is a Vector?

At its core, a vector can be understood from two perspectives:

* **The Physics/Geometric Perspective:** An arrow pointing in space that has both a magnitude (length) and a direction.
* **The Computer Science/ML Perspective:** An ordered list of numbers (an array). In machine learning, a vector represents a single data point. For example, a house could be represented by the vector $\mathbf{x} = [2500, 3, 2]$, representing 2,500 square feet, 3 bedrooms, and 2 bathrooms.

**Notation:**
Vectors are typically written as lowercase bold letters and represented as column matrices:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

Where $n$ is the dimensionality of the vector (e.g., a 3D vector has 3 elements).

## Core Properties of Vectors

Before comparing or transforming vectors, we must understand their fundamental properties and operations.

### Magnitude (The L2 Norm)

The magnitude is the physical length of the vector, calculated using the Pythagorean theorem extended to $n$ dimensions.

$$\Vert{}\mathbf{v}\Vert{}_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}$$

**ML Application:** Used in L2 Regularization (Ridge Regression) to penalize large weights in a neural network, preventing the model from over-relying on any single feature.

### Scalar Multiplication

Multiplying a vector by a single number (a scalar) stretches, shrinks, or flips the vector without changing its fundamental line of direction.

$$c\mathbf{v} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \end{bmatrix}$$

**ML Application:** Scaling the learning rate by the gradient during backpropagation.

### Vector Addition

Adding two vectors together moves point to point (the "parallelogram rule" in geometry). You simply add the corresponding elements.

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$$

**ML Application:** Adding a bias vector to the weighted sum in a neural network layer.

## Vector Dot Product & Cosine Similarity

The dot product measures how much two vectors "point in the same direction." It is the foundation of matrix multiplication and similarity algorithms.

**The Formula:**

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \Vert{}\mathbf{u}\Vert{}_2 \Vert{}\mathbf{v}\Vert{}_2 \cos(\theta)$$

**Cosine Similarity:**

$$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{}_2 \Vert{}\mathbf{v}\Vert{}_2}$$

* **Intuition:** If you have two vectors representing the frequency of words in two documents, their dot product grows larger the more words they share. Dividing by their magnitudes (lengths) gives the Cosine Similarity, bounding the result between $-1$ (opposites) and $1$ (identical directions).
* **ML Application:** Natural Language Processing (NLP) uses cosine similarity to determine if two word embeddings (like "King" and "Queen") have similar semantic meanings. Retrieval-Augmented Generation (RAG) systems use it to find the most relevant document chunks for an LLM prompt.

**Example:** For $\mathbf{u} = [1, 2]$ and $\mathbf{v} = [3, 4]$:

$$\mathbf{u} \cdot \mathbf{v} = (1 \times 3) + (2 \times 4) = 3 + 8 = 11$$

## Matrix-Vector Multiplication

A matrix is a 2D grid of numbers. Matrix-vector multiplication applies a linear transformation (like rotating, scaling, or shearing) to an input vector.

**The Formula:**

$$y_i = \sum_{j=1}^{d} A_{ij} x_j \iff \mathbf{y} = A\mathbf{x}$$

* **Intuition:** Think of the matrix $A$ as a machine. You feed it an input vector $\mathbf{x}$, and it spits out a transformed output vector $\mathbf{y}$.
* **ML Application:** In a neural network layer, $A$ represents the weight matrix and $\mathbf{x}$ represents the input features. Multiplying them calculates the weighted sum for every neuron in the next layer simultaneously. Instead of looping through thousands of neurons, GPUs execute this single matrix operation instantly.

## Eigenvalues and Eigenvectors

When a matrix transforms a space, most vectors get knocked off their original span (direction). Vectors that stay on their original span and only get stretched or squished by the transformation are called eigenvectors.

**The Formula:**

$$A\mathbf{v} = \lambda\mathbf{v} \iff (A - \lambda I)\mathbf{v} = \mathbf{0}$$

* **Intuition:** $\mathbf{v}$ is the eigenvector (the direction), and $\lambda$ is the eigenvalue (the stretching factor). If $\lambda = 2$, the vector gets twice as long. If $\lambda = -1$, it flips in the opposite direction.
* **ML Application:** Used heavily in dimensionality reduction (PCA) to find the axes of greatest variance in a dataset, and for understanding the stability of deep learning optimization by analyzing the eigenvalues of the Hessian matrix.

## Singular Value Decomposition (SVD)

SVD is a factorization technique that decomposes any complex matrix into three simpler, interpretable matrices: a rotation, a scaling, and another rotation. It is one of the most important algorithms in applied linear algebra.

**The Formula:**

$$A = U \Sigma V^T$$

* **Where:**
    * $U$ and $V$ are orthogonal matrices (representing rotations/reflections in high-dimensional space).
    * $\Sigma$ is a diagonal matrix containing singular values $\sigma_i$ (representing stretching/scaling factors).
* **Intuition:** It breaks down a dataset into its most fundamental, independent underlying features, ranked by importance (the magnitude of $\sigma_i$).
* **ML Application:** Principal Component Analysis (PCA) uses SVD to compress data. By dropping the smallest singular values in $\Sigma$, we can reduce a 1,000-feature dataset (like high-res images) down to 10 core features while retaining most of the original mathematical variance.
