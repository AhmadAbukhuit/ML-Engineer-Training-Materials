"""
Retrieval-Augmented Generation (RAG) and Vector Similarity Search.

This module implements:
1. Dense Vector Index: Vectorized exact nearest neighbor search using cosine similarity.
2. Embedding Generation: Normalization and projection into a semantic latent space.
3. Document Chunking & Ingestion: Chunks text and associates unique document IDs and metadata.
4. RAG Pipeline: Ingests documents, retrieves top-k relevant contexts, and constructs
   context-augmented prompts for generative models.
"""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any
import numpy as np


@dataclass
class DocumentChunk:
    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    embedding: np.ndarray | None = None


class SimpleDenseEmbedder:
    """
    Lightweight, reproducible dense text embedder based on subword hashing
    and normalized random projection to d_dimensions.
    Ensures identical text produces identical normalized embedding vectors.
    """

    def __init__(self, embedding_dim: int = 64, random_state: int = 42):
        self.embedding_dim = embedding_dim
        self.vocab_size = 10000
        rng = np.random.default_rng(random_state)
        # Random projection matrix from hash space to latent dense space
        self.projection_matrix = rng.normal(0, 1.0, (self.vocab_size, embedding_dim))

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens if tokens else ["<empty>"]

    def embed(self, text: str) -> np.ndarray:
        """Embeds a single string into a unit-normalized vector."""
        tokens = self._tokenize(text)
        bow = np.zeros(self.vocab_size)
        for token in tokens:
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.vocab_size
            bow[idx] += 1.0

        dense = np.matmul(bow, self.projection_matrix)
        norm = np.linalg.norm(dense)
        if norm > 1e-12:
            dense = dense / norm
        return dense

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch embeds multiple texts into a matrix of shape (N, embedding_dim)."""
        return np.array([self.embed(t) for t in texts])


class VectorIndex:
    """
    In-memory vector store supporting cosine similarity search.
    """

    def __init__(self, metric: str = "cosine"):
        if metric not in ("cosine", "dot"):
            raise ValueError(f"Unsupported metric: {metric}. Choose 'cosine' or 'dot'.")
        self.metric = metric
        self.chunks: list[DocumentChunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Adds document chunks with embeddings to the index."""
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            return

        self.chunks.extend(valid_chunks)
        new_vecs = np.array([c.embedding for c in valid_chunks])
        if self._matrix is None:
            self._matrix = new_vecs
        else:
            self._matrix = np.vstack([self._matrix, new_vecs])

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieves top_k most similar document chunks.

        Returns:
            List of (DocumentChunk, similarity_score) tuples sorted in descending order.
        """
        if self._matrix is None or len(self.chunks) == 0:
            return []

        q = query_vec.copy()
        if self.metric == "cosine":
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-12:
                q = q / q_norm

            matrix_norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            matrix_norms = np.where(matrix_norms < 1e-12, 1e-12, matrix_norms)
            normalized_matrix = self._matrix / matrix_norms
            scores = np.matmul(normalized_matrix, q)
        else:
            scores = np.matmul(self._matrix, q)

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline combining text chunking,
    dense indexing, retrieval, and prompt context synthesis.
    """

    def __init__(self, embedder: SimpleDenseEmbedder | None = None):
        self.embedder = embedder or SimpleDenseEmbedder()
        self.index = VectorIndex(metric="cosine")

    def ingest_documents(
        self,
        docs: list[dict[str, Any]],
        chunk_size: int = 150,
        chunk_overlap: int = 30,
    ) -> int:
        """
        Chunks texts, embeds them, and populates the vector index.

        Args:
            docs: List of dicts containing 'id', 'text', and optional 'metadata'.
            chunk_size: Approximate character length per chunk.
            chunk_overlap: Overlapping character count across chunks.

        Returns:
            Total number of ingested chunks.
        """
        chunks: list[DocumentChunk] = []

        for doc in docs:
            doc_id = doc["id"]
            text = doc["text"]
            meta = doc.get("metadata", {})

            # Chunking with sliding window
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    emb = self.embedder.embed(chunk_text)
                    chunk_meta = dict(meta)
                    chunk_meta["chunk_id"] = f"{doc_id}_{chunk_idx}"
                    chunks.append(
                        DocumentChunk(
                            doc_id=doc_id,
                            text=chunk_text,
                            metadata=chunk_meta,
                            embedding=emb,
                        )
                    )
                    chunk_idx += 1
                if end == len(text):
                    break
                start += max(1, chunk_size - chunk_overlap)

        self.index.add(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        """Finds top_k relevant context chunks for a query."""
        query_vec = self.embedder.embed(query)
        return self.index.search(query_vec, top_k=top_k)

    def build_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Constructs a retrieval-augmented prompt injected with grounding context.
        """
        retrieved = self.retrieve(query, top_k=top_k)
        contexts = [
            f"[{i + 1}] (Source: {item[0].doc_id}) {item[0].text}"
            for i, item in enumerate(retrieved)
        ]
        context_str = "\n".join(contexts) if contexts else "No relevant context found."

        prompt = (
            f"Use the following retrieved context to answer the user query.\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Query: {query}\n\n"
            f"Answer:"
        )
        return prompt
