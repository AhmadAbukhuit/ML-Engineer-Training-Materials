import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transfer_learning import PretrainedBackbone, TransferClassifier


def test_pretrained_backbone_forward_and_freeze():
    backbone = PretrainedBackbone(layer_dims=[8, 16, 4], random_state=42)
    assert backbone.output_dim == 4
    assert len(backbone.layers) == 2

    x = np.random.randn(10, 8)
    out = backbone.forward(x)
    assert out.shape == (10, 4)

    # Test freezing
    backbone.freeze()
    for layer in backbone.layers:
        assert not layer.trainable

    # Test unfreezing
    backbone.unfreeze()
    for layer in backbone.layers:
        assert layer.trainable


def test_transfer_classifier_head_training_and_finetuning():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1.0, (100, 10))
    # Synthetic 2-class classification
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    backbone = PretrainedBackbone(layer_dims=[10, 16, 8], random_state=42)
    initial_backbone_w0 = backbone.layers[0].W.copy()

    classifier = TransferClassifier(backbone=backbone, n_classes=2, random_state=42)

    # Forward check
    logits = classifier.forward(X[:5])
    assert logits.shape == (5, 2)
    probs = classifier.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # Stage 1: Fit head only with frozen backbone
    losses_stage1 = classifier.fit_head_only(X, y, epochs=15, lr=0.1, batch_size=25)
    assert len(losses_stage1) == 15
    assert losses_stage1[-1] < losses_stage1[0]  # Loss decreases

    # Verify backbone weights did not change during Stage 1
    assert np.allclose(backbone.layers[0].W, initial_backbone_w0)

    # Stage 2: Fine-tune backbone and head
    losses_stage2 = classifier.fine_tune(X, y, epochs=10, head_lr=0.05, backbone_lr=0.01, batch_size=25)
    assert len(losses_stage2) == 10

    # Verify backbone weights updated during Stage 2
    assert not np.allclose(backbone.layers[0].W, initial_backbone_w0)

    # Evaluation
    metrics = classifier.evaluate(X, y)
    assert "accuracy" in metrics
    assert "loss" in metrics
    assert metrics["accuracy"] >= 0.60
