"""
Deep Learning Architectures and Computer Vision / NLP Modules.
"""

from .detection_metrics import compute_iou_matrix, non_max_suppression
from .attention import scaled_dot_product_attention, MultiHeadAttention
from .resnet import ResidualBlock
from .training_engine import EarlyStopping, TrainingEngine
from .transfer_learning import DenseLayer, PretrainedBackbone, TransferClassifier
from .rag_vector_search import DocumentChunk, SimpleDenseEmbedder, VectorIndex, RAGPipeline
from .lora import LoRALinear

__all__ = [
    "DenseLayer",
    "DocumentChunk",
    "EarlyStopping",
    "LoRALinear",
    "MultiHeadAttention",
    "PretrainedBackbone",
    "RAGPipeline",
    "ResidualBlock",
    "SimpleDenseEmbedder",
    "TrainingEngine",
    "TransferClassifier",
    "VectorIndex",
    "compute_iou_matrix",
    "non_max_suppression",
    "scaled_dot_product_attention",
]

