import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_vector_search import DocumentChunk, RAGPipeline, SimpleDenseEmbedder, VectorIndex


def test_simple_dense_embedder():
    embedder = SimpleDenseEmbedder(embedding_dim=32, random_state=42)
    vec1 = embedder.embed("Gradient descent optimizes model weights.")
    vec2 = embedder.embed("Gradient descent optimizes model weights.")
    vec3 = embedder.embed("Cats and dogs are common domestic pets.")

    # Consistency check
    assert np.allclose(vec1, vec2)
    assert vec1.shape == (32,)

    # L2-normalization check
    assert np.isclose(np.linalg.norm(vec1), 1.0)
    assert np.isclose(np.linalg.norm(vec3), 1.0)

    # Cosine similarity check: same text is 1.0, unrelated text is lower
    cos_same = float(np.dot(vec1, vec2))
    cos_diff = float(np.dot(vec1, vec3))
    assert np.isclose(cos_same, 1.0)
    assert cos_diff < cos_same


def test_vector_index_search():
    index = VectorIndex(metric="cosine")
    c1 = DocumentChunk("doc1", "Linear regression fits a hyperplane.", embedding=np.array([1.0, 0.0, 0.0]))
    c2 = DocumentChunk("doc2", "Neural networks use backpropagation.", embedding=np.array([0.0, 1.0, 0.0]))
    c3 = DocumentChunk("doc3", "Decision trees split on information gain.", embedding=np.array([0.0, 0.0, 1.0]))

    index.add([c1, c2, c3])

    # Search for vector identical to c2
    results = index.search(np.array([0.0, 1.0, 0.0]), top_k=2)
    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk.doc_id == "doc2"
    assert np.isclose(top_score, 1.0)


def test_rag_pipeline_end_to_end():
    rag = RAGPipeline(embedder=SimpleDenseEmbedder(embedding_dim=128, random_state=42))

    docs = [
        {
            "id": "ml_glossary",
            "text": "Overfitting occurs when a model learns noise in training data and fails to generalize.",
        },
        {
            "id": "db_overview",
            "text": "PostgreSQL is an advanced relational database management system using SQL.",
        },
    ]

    total_chunks = rag.ingest_documents(docs, chunk_size=100, chunk_overlap=20)
    assert total_chunks >= 2

    # Query for overfitting
    retrieved = rag.retrieve("Why does overfitting happen in training?", top_k=1)
    assert len(retrieved) == 1
    chunk, _ = retrieved[0]
    assert chunk.doc_id == "ml_glossary"

    # Build prompt
    prompt = rag.build_prompt("Explain overfitting.", top_k=1)
    assert "Context:" in prompt
    assert "ml_glossary" in prompt
    assert "User Query: Explain overfitting." in prompt
