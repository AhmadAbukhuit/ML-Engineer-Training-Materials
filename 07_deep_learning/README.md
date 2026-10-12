# 🧠 Phase 4: Deep Learning & Vision/NLP Architectures

Modern Machine Learning Engineering demands a rigorous understanding of deep neural network architectures—from spatial feature extraction in Convolutional Neural Networks (CNNs) and spatial bounding-box reasoning in YOLO, to sequence transduction and self-attention in Transformers.

---

## 🎯 Learning Objectives

By the end of this phase, you will:

1. **Understand Deep Learning Dynamics:** Master backpropagation over deep computational graphs, vanishing/exploding gradients, weight initializations (He/Kaiming, Xavier/Glorot), and normalization techniques (Batch Normalization, Layer Normalization).
2. **Master Computer Vision (CNNs & Residual Networks):** Implement convolutional feature extractors, understand 2D receptive fields, and build residual skip-connections (`F(x) + x`) to train deep networks without degradation.
3. **Master Object Detection (YOLO Mechanics):** Implement vectorized Intersection over Union (IoU) and greedy Non-Maximum Suppression (NMS) to eliminate duplicate candidate bounding boxes.
4. **Master NLP & Transformer Attention:** Formulate Scaled Dot-Product Self-Attention and Multi-Head Attention mechanisms from scratch, understanding how queries, keys, and values route contextual information.
5. **Build Modular Training Engines:** Construct production-grade training loops with validation, learning rate scheduling (Cosine Annealing), early stopping, and checkpoint serialization.

---

## 📚 Curated Study Resources

### 📖 Essential Papers & Guides

* **[Deep Learning Book](https://www.deeplearningbook.org/)** (Goodfellow, Bengio, Courville) — Chapters 6–10 cover MLPs, Optimization, CNNs, and Sequence Modeling.
* **[Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)** (He et al., 2015) — Foundational paper introducing skip connections.
* **[Attention Is All You Need (Transformers)](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017) — The landmark paper replacing recurrence with self-attention.
* **[You Only Look Once (YOLO)](https://arxiv.org/abs/1506.02640)** (Redmon et al., 2016) — Real-time unified object detection framework.

### 🎥 Video Lectures

* **Andrej Karpathy - Neural Networks: Zero to Hero:** Deep dive into backpropagation, language modeling (Makemore), and building GPT from scratch.
* **Stanford CS231n - Convolutional Neural Networks for Visual Recognition:** The gold standard university curriculum for computer vision.

---

## 🔬 Core Architectural Formats & Equations

### 1. Residual Networks (ResNet)

Standard feedforward layers suffer from degradation when stacked too deeply. Residual blocks solve this by learning residual mappings:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathcal{W}_s \mathbf{x}$$

* When dimensions match, $\mathcal{W}_s = I$ (identity shortcut), allowing gradients to backpropagate uninterrupted:
  $$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + I \right)$$
* Even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}} \to \mathbf{0}$, the identity term $I$ preserves gradient flow back to early layers.

### 2. Object Detection Metrics: IoU & NMS

* **Intersection over Union (IoU):**
  $$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$
* **Non-Maximum Suppression (NMS) Algorithm:**
  1. Sort all bounding boxes by predicted confidence score $s_i$.
  2. Select the box $M$ with highest score and add it to final output.
  3. Discard any remaining box $B$ where $\text{IoU}(M, B) \ge \text{threshold}$ (typically 0.5).
  4. Repeat until no candidate boxes remain.

### 3. Scaled Dot-Product Self-Attention

Given query matrix $Q$, key matrix $K$, and value matrix $V$ with key dimension $d_k$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

Where $M$ is an optional causal mask (for autoregressive decoders) setting future positions to $-\infty$.

* **Multi-Head Attention (MHA):**
  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
  $$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

---

## 🛠️ Practical Coding Tasks

Complete these implementation tasks in `04_deep_learning/src/` and verify your solutions with tests in `04_deep_learning/tests/`.

---

### Task 4.1: Vectorized IoU & Non-Maximum Suppression (NMS)

* **File:** `src/detection_metrics.py`
* **Requirements:**
  1. Implement `compute_iou_matrix(boxes_a, boxes_b)` for batch pairwise IoU calculation between $N$ and $M$ boxes in pure NumPy.
  2. Implement `non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.25)` to eliminate overlapping candidate boxes.
* **Deliverable:** Unit tests validating suppression accuracy on synthetic multi-box cluster scenarios.

---

### Task 4.2: Scaled Dot-Product & Multi-Head Self-Attention

* **File:** `src/attention.py`
* **Requirements:**
  1. Implement `scaled_dot_product_attention(Q, K, V, mask=None)` with $\frac{1}{\sqrt{d_k}}$ scaling and numerically stable softmax.
  2. Implement `MultiHeadAttention` class projecting inputs through learned linear projections $W^Q, W^K, W^V, W^O$.
  3. Support both unmasked (encoder) and causally masked (decoder) forward passes.
* **Deliverable:** Unit tests verifying output tensor shape preservation $(B, T, D)$ and attention weight normalization.

---

### Task 4.3: Residual Block Architecture

* **File:** `src/resnet.py`
* **Requirements:**
  1. Implement `ResidualBlock` featuring two convolutional/linear transformations, batch/layer normalization, and an additive skip connection.
  2. Implement projection shortcuts when input and output feature dimensions differ.
* **Deliverable:** Unit tests demonstrating gradient flow and dimension matching.

---

### Task 4.4: Reusable Deep Learning Training Engine

* **File:** `src/training_engine.py`
* **Requirements:**
  1. Create a `TrainingEngine` class encapsulating train step, validation step, and epoch progression.
  2. Implement early stopping monitoring validation loss with a patience counter.
  3. Support learning rate scheduling and model weight checkpoint serialization.
* **Deliverable:** Standalone demonstration script showing converging training curves.

---

### Task 4.5: Transfer Learning & Backbone Freezing

* **File:** `src/transfer_learning.py`
* **Requirements:**
  1. Implement `PretrainedBackbone` feature extractor with layer-wise freezing and unfreezing mechanics.
  2. Implement `TransferClassifier` replacing target heads and executing two-stage fine-tuning (head training followed by discriminative learning rate fine-tuning).
* **Deliverable:** Unit tests verifying gradient non-propagation during frozen stages and loss convergence.

---

### Task 4.6: Dense Vector Search & RAG Pipeline

* **File:** `src/rag_vector_search.py`
* **Requirements:**
  1. Implement `SimpleDenseEmbedder` mapping texts to normalized latent dense vectors via deterministic hashing and random projection.
  2. Implement `VectorIndex` for exact cosine similarity nearest-neighbor search.
  3. Implement `RAGPipeline` supporting text chunking, sliding-window overlap, semantic retrieval, and prompt context synthesis.
* **Deliverable:** Unit tests validating retrieval recall and prompt grounding.

---

### Task 4.7: Parameter-Efficient Fine-Tuning (LoRA)

* **File:** `src/lora.py`
* **Requirements:**
  1. Implement `LoRALinear` decomposing weight updates into $\Delta W = \frac{\alpha}{r} (A \cdot B)$ while keeping base weights $W_0$ frozen.
  2. Implement backward pass updating only adapter matrices $A$ and $B$.
  3. Implement zero-overhead inference weight merging (`merge_weights` / `unmerge_weights`).
* **Deliverable:** Unit tests demonstrating 90%+ parameter reduction and output identity post-merging.

---

## 📁 Directory Structure for Phase 4

```text
04_deep_learning/
├── README.md
├── 01_cnn_and_residual_architectures.md
├── 02_transformers_and_attention.md
├── 03_lora_and_rag.md
├── notebooks/
│   └── 01_deep_learning_architectures.ipynb
├── weights/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── resnet.py
│   ├── attention.py
│   ├── detection_metrics.py
│   ├── training_engine.py
│   ├── transfer_learning.py
│   ├── rag_vector_search.py
│   └── lora.py
└── tests/
    ├── __init__.py
    ├── test_resnet.py
    ├── test_attention.py
    ├── test_detection_metrics.py
    ├── test_transfer_learning.py
    ├── test_rag.py
    └── test_lora.py
```
