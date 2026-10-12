import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_retrieval import HybridSearchEngine, reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    list1 = ["doc_A", "doc_B", "doc_C"]
    list2 = ["doc_B", "doc_A", "doc_D"]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    top_doc, top_score = fused[0]
    second_doc, _ = fused[1]

    # doc_A has ranks 1 and 2: 1/61 + 1/62
    # doc_B has ranks 2 and 1: 1/62 + 1/61 -> identical score to doc_A
    assert {top_doc, second_doc} == {"doc_A", "doc_B"}
    assert top_score > 0.0


def test_hybrid_search_engine_ranking():
    doc_ids = ["doc_1", "doc_2", "doc_3"]
    documents = [
        "deep neural networks and transformer self-attention",
        "random forests and gradient boosted decision trees",
        "database query optimization with postgresql and duckdb",
    ]
    # Synthetic 4-dimensional embeddings
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    engine = HybridSearchEngine(k_rrf=60).index_corpus(doc_ids, documents, embeddings)

    # Search query combining keywords and vector
    q_text = "transformer neural networks"
    q_vector = np.array([0.9, 0.1, 0.0, 0.0])

    results = engine.hybrid_search(q_text, q_vector, top_k=2)
    top_doc_id, _ = results[0]

    assert top_doc_id == "doc_1"
