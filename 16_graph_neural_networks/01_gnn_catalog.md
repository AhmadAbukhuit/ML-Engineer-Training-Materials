# Complete Catalog of Graph Machine Learning & Graph Neural Networks

Relational domains (molecules, social networks, knowledge graphs, citation networks) cannot be represented by Euclidean grid architectures (CNNs) or sequential models (RNNs/Transformers). Graph Neural Networks (GNNs) operate on non-Euclidean topological structures by iteratively exchanging and aggregating local information across node neighborhoods.

---

## 1. The Message Passing Framework (MPNN)

Gilmer et al. (2017) formulated the overarching mathematical framework unifying spatial graph deep learning into three distinct phases per layer $l$:

### 1.1 Message Phase
Nodes send directed messages $m_{ij}^{(l)}$ along incident edges $(j, i) \in \mathcal{E}$:

$$m_{ij}^{(l)} = M^{(l)}\left( h_i^{(l)}, \, h_j^{(l)}, \, e_{ij} \right)$$

where $M^{(l)}$ is a parameterized neural network (e.g. MLP) and $e_{ij}$ represents optional edge attributes.

### 1.2 Aggregation Phase
Incoming messages from all incoming neighbors $\mathcal{N}(i)$ are aggregated using a **permutation-invariant** operator $\bigoplus$:

$$m_i^{(l)} = \bigoplus_{j \in \mathcal{N}(i)} m_{ij}^{(l)}, \quad \bigoplus \in \{\sum, \, \text{mean}, \, \max\}$$

Permutation invariance guarantees that graph isomorphism is respected: reordering the indices of adjacent nodes does not alter the aggregated message.

### 1.3 Update Phase
The central node updates its latent representation by combining its previous state with the aggregated neighborhood context:

$$h_i^{(l+1)} = U^{(l)}\left( h_i^{(l)}, \, m_i^{(l)} \right)$$

---

## 2. Graph Convolutional Networks (GCN)

Kipf & Welling (2017) derived GCN as a localized, first-order Chebyshev polynomial approximation of spectral graph convolutions.

### 2.1 The Renormalization Trick
Standard graph adjacency $A$ does not include self-connections, which causes node representations to omit their own prior features during neighborhood aggregation. Furthermore, multiplication by $A$ scales feature magnitudes proportionally to node degrees.

To resolve these issues, the renormalization trick computes:
$$\tilde{A} = A + I_N, \quad \tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$$
$$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

where $\hat{A}_{ij} = \frac{1}{\sqrt{\tilde{D}_{ii} \tilde{D}_{jj}}}$ if $(i, j) \in \mathcal{E}$ or $i = j$.

### 2.2 Layer Propagation Rule
The vectorized propagation across the entire graph is:

$$H^{(l+1)} = \sigma\left( \hat{A} H^{(l)} W^{(l)} \right)$$

where $H^{(0)} = X \in \mathbb{R}^{N \times F}$ is the initial node feature matrix, $W^{(l)}$ is the layer weight matrix, and $\sigma(\cdot)$ is an element-wise activation function (e.g. ReLU).

---

## 3. Graph Attention Networks (GAT)

While GCN assigns static, topology-dependent scalar weights $\hat{A}_{ij}$ to edges, Veličković et al. (2018) introduced **Graph Attention Networks (GAT)** to dynamically attend over neighbors conditioned on both node features.

### 3.1 Self-Attention Mechanism
For node pair $(i, j)$ with linear projection $z_i = W h_i$:

$$e_{ij} = \text{LeakyReLU}\left( a_{\text{src}}^T z_i + a_{\text{dst}}^T z_j \right)$$

The attention coefficients are normalized across all one-hop neighbors via masked softmax:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

### 3.2 Multi-Head Attention Aggregation
To stabilize the learning process, $K$ independent attention heads compute projections concurrently:
- **Hidden Layers (Concatenation):**
  $$h_i^{(l+1)} = \Vert_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j^{(l)} \right)$$
- **Output Layer (Averaging):**
  $$h_i^{(L)} = \sigma\left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j^{(L-1)} \right)$$
