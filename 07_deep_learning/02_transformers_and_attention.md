# Transformers & Scaled Dot-Product Attention

The Transformer architecture eliminated sequential recurrence (RNNs/LSTMs) in favor of **Self-Attention**, enabling massive parallelization across GPUs and capturing long-range dependencies regardless of token distance.

---

## 1. Information Retrieval Analogy: Queries, Keys, and Values

The core attention mechanism is modeled after information retrieval systems:
* **Query ($Q$):** What the current token is looking for (e.g., "I need the subject of this verb").
* **Key ($K$):** What each token contains or advertises (e.g., "I am a noun phrase").
* **Value ($V$):** The actual informational content passed forward if the query matches the key.

---

## 2. Scaled Dot-Product Attention

Given matrices $Q \in \mathbb{R}^{T \times d_k}$, $K \in \mathbb{R}^{T \times d_k}$, and $V \in \mathbb{R}^{T \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

### Why the Scaling Factor $\frac{1}{\sqrt{d_k}}$?
For large key dimensions $d_k$, the dot products $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ grow large in magnitude. 
Assuming components of $q$ and $k$ are independent random variables with zero mean and unit variance:

$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k$$

Without scaling by $\frac{1}{\sqrt{d_k}}$, the variance equals $d_k$, pushing the softmax function into regions with near-zero gradients (saturation), which stalls learning during backpropagation.

---

## 3. Multi-Head Attention (MHA)

Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, Multi-Head Attention projects the queries, keys, and values $h$ times with different learned linear projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### Why Multi-Head?
A single attention head tends to average out multiple semantic relationships. Multi-Head Attention allows the model to simultaneously attend to information from different representation subspaces at different positions (e.g., one head tracks syntactic agreement, another tracks semantic coreference).

---

## 4. Positional Encodings

Because self-attention operations are permutation-equivariant (they do not inherently know the order of words in a sentence), explicit positional information must be injected into the input embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

This allows the model to easily learn relative positions because for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.
