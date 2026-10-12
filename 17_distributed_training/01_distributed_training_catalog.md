# Complete Catalog of Distributed Deep Learning & Parallelism

Training multi-billion parameter models exceeds the compute and memory limits of single accelerators. Distributed training requires coordinated communication topologies, memory state sharding, and numerical precision management.

---

## 1. High-Performance Collective Communication: Ring All-Reduce

Standard Distributed Data Parallel (DDP) replicates model parameters across $P$ workers. During the backward pass, workers compute local gradients $g_r$ concurrently. Before optimizer updates, workers must calculate the global sum $\sum_{r=0}^{P-1} g_r$ across all devices.

### 1.1 Naive Parameter Server Bottleneck

In a client-server architecture, each worker sends its full tensor of size $N$ to a master node. The master node becomes a catastrophic communication bottleneck, handling $O(P \cdot N)$ network traffic.

### 1.2 The Ring All-Reduce Topology (Patarasuk & Yuan, 2009)

In a ring topology, each rank $r \in \{0, \dots, P-1\}$ sends data only to its logical neighbor $(r + 1) \bmod P$ and receives data only from $(r - 1) \bmod P$. The tensor of size $N$ is divided into $P$ equal chunks.

The algorithm runs in two distinct phases:

1. **Scatter-Reduce ($P - 1$ steps):**
   - At step $k$, rank $r$ sends chunk $(r - k) \bmod P$ and receives chunk $(r - k - 1) \bmod P$, adding received elements to its local chunk.
   - After $P - 1$ steps, each rank holds the fully reduced sum of exactly one chunk of size $N / P$.
2. **All-Gather ($P - 1$ steps):**
   - Each rank sends its fully reduced chunk around the ring for $P - 1$ steps.
   - After $P - 1$ steps, every rank holds the complete elementwise sum of the entire tensor.

### 1.3 Communication Volume Invariant

The total data transmitted per rank across both phases is:

$$\text{Volume per rank} = 2 \left( \frac{P - 1}{P} \right) N$$

As cluster size $P \to \infty$, the data volume per rank approaches $2N$, completely independent of the number of GPUs in the cluster!

---

## 2. Zero Redundancy Optimizer (ZeRO) Memory Partitioning

During mixed-precision Adam training of a model with $\Psi$ parameters, static GPU memory is heavily dominated by redundant optimizer states rather than model weights.

### 2.1 Baseline DDP Memory Breakdown (16 Bytes / Parameter)

- **Model Parameters (FP16/BF16):** $2\Psi$ bytes
- **Gradients (FP16/BF16):** $2\Psi$ bytes
- **Adam Optimizer States (FP32):**
  - Master Weights (FP32): $4\Psi$ bytes
  - Momentum $m$ (FP32): $4\Psi$ bytes
  - Variance $v$ (FP32): $4\Psi$ bytes
  - Subtotal: $12\Psi$ bytes
- **Total Static Memory:** $2\Psi + 2\Psi + 12\Psi = 16\Psi$ bytes per GPU.

### 2.2 ZeRO Stages (Rajbhandari et al., 2020)

Across $N_d$ parallel data workers:

- **ZeRO Stage 1 ($P_{\text{os}}$ - Optimizer State Partitioning):**
  - Shards optimizer states across workers: memory footprint becomes $4\Psi + \frac{12\Psi}{N_d}$ bytes. Yields up to $4\times$ memory reduction with zero communication overhead.
- **ZeRO Stage 2 ($P_{\text{os}+g}$ - Gradient Partitioning):**
  - Shards gradients as they are computed: memory footprint becomes $2\Psi + \frac{14\Psi}{N_d}$ bytes. Yields up to $8\times$ memory reduction with zero communication overhead.
- **ZeRO Stage 3 ($P_{\text{os}+g+p}$ - Parameter Partitioning):**
  - Shards model parameters across workers; workers broadcast parameters on-demand during forward and backward passes: memory footprint becomes $\frac{16\Psi}{N_d}$ bytes. Yields linear $N_d \times$ memory scaling.

---

## 3. Mixed Precision Numerical Stability: Dynamic Loss Scaling

16-bit floating point representations (IEEE 754 FP16) have a limited dynamic range $[5.96 \times 10^{-8}, 65504]$.

### 3.1 Gradient Underflow Problem

During backpropagation, weight gradients frequently have magnitudes below $10^{-5}$. In FP16, these underflow to zero, stalling model convergence.

### 3.2 Dynamic Loss Scaling Solution

1. Multiply loss by scaling factor $S$ (e.g. $2^{16} = 65536$) before backprop: $\mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}$.
2. Gradients are shifted into the representable range of FP16: $g_{\text{scaled}} = S \cdot \nabla_W \mathcal{L}$.
3. Check for numerical overflow ($\text{Inf}$ / $\text{NaN}$):
   - If overflow is detected: discard the mini-batch update and back off scale factor $S \leftarrow S \times 0.5$.
   - If all gradients are finite: unscale gradients $g = \frac{1}{S} g_{\text{scaled}}$ and apply optimizer step. If $N$ consecutive steps succeed without overflow, grow scale factor $S \leftarrow S \times 2.0$.
