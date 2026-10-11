"""Semi-Supervised Learning Module: Self-Training, Graph Propagation, and Consistency Regularization."""

from .consistency_regularization import consistency_loss, rampup_weight, sharpen
from .label_propagation import LabelPropagation
from .pseudo_labeling import SelfTrainingClassifier

__all__ = [
    "LabelPropagation",
    "SelfTrainingClassifier",
    "consistency_loss",
    "rampup_weight",
    "sharpen",
]
