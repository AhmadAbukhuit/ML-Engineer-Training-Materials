"""Tests for Implicit Alternating Least Squares (ALS) Recommender."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from als_recommender import ImplicitALS


def test_implicit_als_fit_and_shapes():
    R = np.array([
        [5, 0, 0, 1],
        [0, 3, 2, 0],
        [4, 0, 0, 2],
        [0, 0, 4, 3],
    ], dtype=float)

    model = ImplicitALS(n_factors=4, alpha=10.0, regularization=0.01, n_iterations=10, random_state=42)
    model.fit(R)

    assert model.user_factors is not None
    assert model.item_factors is not None
    assert model.user_factors.shape == (4, 4)
    assert model.item_factors.shape == (4, 4)


def test_implicit_als_recommend_ordering():
    # User 0 likes item 0 heavily, user 1 likes item 1 heavily
    R = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ])

    model = ImplicitALS(n_factors=4, alpha=20.0, regularization=0.01, n_iterations=15, random_state=42)
    model.fit(R)

    recs_user0 = model.recommend(0, top_k=3)
    assert len(recs_user0) == 3
    # Top recommendation for user 0 should be item 0
    assert recs_user0[0][0] == 0
    # Recs should be sorted in descending order of score
    scores = [s for _, s in recs_user0]
    assert scores == sorted(scores, reverse=True)


def test_implicit_als_predict_score():
    R = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    model = ImplicitALS(n_factors=2, n_iterations=5, random_state=42)
    model.fit(R)
    score = model.predict_score(0, 0)
    assert isinstance(score, float)
