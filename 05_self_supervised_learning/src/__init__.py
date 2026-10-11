"""Self-Supervised Learning Module: Contrastive Learning, InfoNCE, and Autoencoders."""

from .autoencoder import DeepAutoencoder
from .contrastive_loss import NTXentLoss, compute_alignment_and_uniformity
from .simclr_framework import ProjectionHead, SimCLRFramework

__all__ = [
    "DeepAutoencoder",
    "NTXentLoss",
    "ProjectionHead",
    "SimCLRFramework",
    "compute_alignment_and_uniformity",
]
