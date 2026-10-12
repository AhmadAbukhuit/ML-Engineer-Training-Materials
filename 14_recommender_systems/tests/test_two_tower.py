"""Tests for Two-Tower Retrieval Model."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from two_tower import TwoTowerModel


def test_two_tower_embedding_normalization():
    model = TwoTowerModel(user_dim=10, item_dim=12, embed_dim=8, random_state=42)

    user_batch = np.random.randn(5, 10)
    item_batch = np.random.randn(5, 12)

    u_emb = model.encode_users(user_batch)
    i_emb = model.encode_items(item_batch)

    assert u_emb.shape == (5, 8)
    assert i_emb.shape == (5, 8)

    # L2 norms should be approximately 1
    u_norms = np.linalg.norm(u_emb, axis=1)
    i_norms = np.linalg.norm(i_emb, axis=1)
    assert np.allclose(u_norms, 1.0, atol=1e-5)
    assert np.allclose(i_norms, 1.0, atol=1e-5)


def test_two_tower_training_step():
    model = TwoTowerModel(user_dim=6, item_dim=6, embed_dim=4, learning_rate=0.1, random_state=42)
    user_batch = np.eye(4, 6)
    item_batch = np.eye(4, 6)

    initial_loss = model.train_step(user_batch, item_batch)
    for _ in range(20):
        final_loss = model.train_step(user_batch, item_batch)

    assert final_loss < initial_loss


def test_two_tower_candidate_retrieval():
    model = TwoTowerModel(user_dim=4, item_dim=4, embed_dim=4, random_state=42)

    user_vec = np.array([1.0, 0.0, 0.0, 0.0])
    item_embs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ])

    top_idx, scores = model.retrieve_candidates(user_vec, item_embs, top_k=2)
    assert len(top_idx) == 2
    assert len(scores) == 2
    assert scores[0] >= scores[1]
