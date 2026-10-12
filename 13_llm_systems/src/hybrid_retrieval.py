"""Hybrid Search and Reciprocal Rank Fusion (RRF) from First Principles."""

from collections import defaultdict
import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion (RRF) scores across multiple ranked search result lists.

    Formula:
        RRF_Score(d) = sum_{m in ranked_lists} 1 / (k + rank_m(d))

    Parameters
    ----------
    ranked_lists : list of list of str
        List containing ordered document IDs from different retrieval algorithms.
    k : int
        RRF smoothing constant (standard default is 60).

    Returns
    -------
    list of (doc_id, score) sorted descending by RRF score.
    """
    scores: dict[str, float] = defaultdict(float)

    for rank_list in ranked_lists:
        for rank_idx, doc_id in enumerate(rank_list):
            # 1-indexed rank
            rank = rank_idx + 1
            scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_results


class HybridSearchEngine:
    """Hybrid Retrieval combining Lexical TF-IDF and Dense Vector Embeddings via RRF."""

    def __init__(self, k_rrf: int = 60) -> None:
        self.k_rrf = k_rrf
        self.doc_ids: list[str] = []
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.vocab: dict[str, int] = {}
        self.tfidf_matrix: np.ndarray | None = None

    def index_corpus(
        self, doc_ids: list[str], documents: list[str], embeddings: np.ndarray
    ) -> "HybridSearchEngine":
        """Index textual documents and dense vectors."""
        self.doc_ids = doc_ids
        self.documents = documents
        self.embeddings = np.asarray(embeddings, dtype=np.float64)

        # Build simple TF-IDF lexical index
        tokenized_docs = [doc.lower().split() for doc in documents]
        all_words = {w for doc in tokenized_docs for w in doc}
        self.vocab = {word: i for i, word in enumerate(sorted(all_words))}

        n_docs = len(documents)
        n_vocab = len(self.vocab)
        tf_matrix = np.zeros((n_docs, n_vocab), dtype=np.float64)

        for i, tokens in enumerate(tokenized_docs):
            for t in tokens:
                if t in self.vocab:
                    tf_matrix[i, self.vocab[t]] += 1.0
            total_tokens = len(tokens)
            if total_tokens > 0:
                tf_matrix[i] /= total_tokens

        # IDF
        doc_freq = np.sum(tf_matrix > 0, axis=0)
        idf = np.log((n_docs + 1.0) / (doc_freq + 1.0)) + 1.0
        self.tfidf_matrix = tf_matrix * idf

        return self

    def lexical_search(self, query: str, top_n: int = 5) -> list[str]:
        """Rank documents by lexical TF-IDF cosine similarity."""
        assert self.tfidf_matrix is not None
        q_tokens = query.lower().split()
        q_vec = np.zeros(len(self.vocab), dtype=np.float64)
        for t in q_tokens:
            if t in self.vocab:
                q_vec[self.vocab[t]] += 1.0

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        scores = np.dot(self.tfidf_matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.doc_ids[i] for i in top_indices]

    def dense_search(self, query_vector: np.ndarray, top_n: int = 5) -> list[str]:
        """Rank documents by dense vector cosine similarity."""
        assert self.embeddings is not None
        q = np.asarray(query_vector, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        emb_normed = self.embeddings / np.maximum(emb_norms, 1e-12)

        scores = np.dot(emb_normed, q)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.doc_ids[i] for i in top_indices]

    def hybrid_search(
        self, query_text: str, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Combine lexical and dense rankings using Reciprocal Rank Fusion."""
        lexical_ranking = self.lexical_search(query_text, top_n=len(self.doc_ids))
        dense_ranking = self.dense_search(query_vector, top_n=len(self.doc_ids))

        fused = reciprocal_rank_fusion([lexical_ranking, dense_ranking], k=self.k_rrf)
        return fused[:top_k]
