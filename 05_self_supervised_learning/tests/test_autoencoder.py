import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autoencoder import DeepAutoencoder


def test_autoencoder_forward_and_shapes():
    model = DeepAutoencoder(input_dim=16, hidden_dim=8, latent_dim=4, random_state=42)

    X = np.random.randn(20, 16)
    X_recon, z, _ = model.forward(X)

    assert X_recon.shape == (20, 16)
    assert z.shape == (20, 4)

    # Encode and decode separately
    z_latent = model.encode(X)
    assert z_latent.shape == (20, 4)
    X_out = model.decode(z_latent)
    assert X_out.shape == (20, 16)


def test_autoencoder_training_loss_decreases():
    rng = np.random.default_rng(42)
    # Synthetic correlated data
    base = rng.normal(size=(50, 4))
    mixing = rng.normal(size=(4, 12))
    X = base @ mixing

    model = DeepAutoencoder(
        input_dim=12,
        hidden_dim=16,
        latent_dim=4,
        noise_factor=0.01,
        learning_rate=0.01,
        random_state=42,
    )

    initial_loss = model.train_step(X)
    for _ in range(40):
        final_loss = model.train_step(X)

    assert final_loss < initial_loss
