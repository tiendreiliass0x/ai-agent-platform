"""
Hybrid Retriever - Dense + Sparse Retrieval

Combines vector similarity search (dense) with BM25 keyword search (sparse)
for optimal retrieval accuracy.

Key Innovation:
- Dense embeddings capture semantic meaning ("AI agents" matches "intelligent systems")
- Sparse BM25 captures exact keywords ("pricing" must appear in text)
- Reciprocal Rank Fusion (RRF) combines both without score normalization issues
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import Counter
import math


@dataclass
class RetrievalResult:
    """A single retrieval result with scoring"""
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_method: str = "hybrid"  # "dense", "sparse", or "hybrid"
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    chunk_id: Optional[str] = None


class BM25:
    """
    BM25 (Best Matching 25) - Sparse keyword retrieval.

    Industry standard for keyword matching, better than TF-IDF for retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with tuning parameters.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Length normalization parameter (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.corpus_size = 0
        self.avgdl = 0  # Average document length
        self.doc_freqs = []  # Document frequencies per term
        self.idf = {}  # Inverse document frequency
        self.doc_lengths = []

    def fit(self, corpus: List[str]) -> None:
        """
        Build BM25 index from corpus.

        Args:
            corpus: List of documents (strings)
        """
        self.corpus_size = len(corpus)
        self.corpus = corpus

        # Tokenize and compute document frequencies
        doc_term_freqs = []
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            # Term frequencies in this document
            term_freqs = Counter(tokens)
            doc_term_freqs.append(term_freqs)

            # Update document frequencies (how many docs contain each term)
            for term in set(tokens):
                if term not in self.idf:
                    self.idf[term] = 0
                self.idf[term] += 1

        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = doc_term_freqs

        # Compute IDF scores
        for term, df in self.idf.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search corpus using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_index, score) tuples, sorted by score desc
        """
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size)

        for doc_id in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lengths[doc_id]
            term_freqs = self.doc_freqs[doc_id]

            for token in query_tokens:
                if token not in self.idf:
                    continue

                idf = self.idf[token]
                tf = term_freqs.get(token, 0)

                # BM25 score for this term
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)

            scores[doc_id] = score

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be improved with proper tokenizer)"""
        # Lowercase and split on whitespace/punctuation
        text = text.lower()
        # Simple split (could use nltk or spacy for better tokenization)
        tokens = text.split()
        # Remove punctuation
        tokens = [''.join(c for c in token if c.isalnum()) for token in tokens]
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        return tokens


class HybridRetriever:
    """
    Hybrid retrieval combining dense vector search and sparse BM25.

    Strategy:
    1. Dense search: Find semantically similar content via embeddings
    2. Sparse search: Find keyword matches via BM25
    3. Fusion: Combine using Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize hybrid retriever.

        Args:
            corpus: List of text documents
            metadata: Optional metadata for each document
            embeddings: Optional pre-computed embeddings (shape: [n_docs, embedding_dim])
            k1: BM25 term frequency saturation
            b: BM25 length normalization
        """
        self.corpus = corpus or []
        self.metadata = metadata or []
        self.embeddings = embeddings

        # Initialize BM25
        self.bm25 = BM25(k1=k1, b=b)
        if self.corpus:
            self.bm25.fit(self.corpus)

    def index_corpus(
        self,
        corpus: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """
        Index a corpus for hybrid retrieval.

        Args:
            corpus: List of text documents
            metadata: Optional metadata for each document
            embeddings: Optional pre-computed embeddings
        """
        self.corpus = corpus
        self.metadata = metadata or [{} for _ in corpus]
        self.embeddings = embeddings

        # Build BM25 index
        self.bm25.fit(corpus)

    def retrieve_dense(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Dense vector retrieval using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results

        Returns:
            List of retrieval results sorted by similarity
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Compute cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)

        # Avoid division by zero
        similarities = np.zeros(len(self.embeddings))
        for i in range(len(self.embeddings)):
            if doc_norms[i] > 0 and query_norm > 0:
                similarities[i] = np.dot(query_embedding, self.embeddings[i]) / (query_norm * doc_norms[i])

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            if similarities[idx] <= 0:
                continue

            result = RetrievalResult(
                text=self.corpus[idx],
                score=float(similarities[idx]),
                metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                retrieval_method="dense",
                dense_score=float(similarities[idx]),
                chunk_id=str(idx)
            )
            results.append(result)

        return results

    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Sparse keyword retrieval using BM25.

        Args:
            query: Search query text
            top_k: Number of results

        Returns:
            List of retrieval results sorted by BM25 score
        """
        bm25_results = self.bm25.search(query, top_k=top_k)

        results = []
        for doc_id, score in bm25_results:
            result = RetrievalResult(
                text=self.corpus[doc_id],
                score=score,
                metadata=self.metadata[doc_id] if doc_id < len(self.metadata) else {},
                retrieval_method="sparse",
                sparse_score=score,
                chunk_id=str(doc_id)
            )
            results.append(result)

        return results

    def retrieve_hybrid(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        alpha: float = 0.5,
        use_rrf: bool = True,
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining dense and sparse methods.

        Args:
            query: Search query text
            query_embedding: Optional query embedding (if None, only sparse search)
            top_k: Number of final results
            alpha: Balance between dense (1.0) and sparse (0.0) when not using RRF
            use_rrf: Use Reciprocal Rank Fusion (recommended)
            rrf_k: RRF constant (typically 60)

        Returns:
            List of retrieval results with hybrid scoring
        """
        # Get results from both methods
        sparse_results = self.retrieve_sparse(query, top_k=top_k * 2)

        dense_results = []
        if query_embedding is not None and self.embeddings is not None:
            dense_results = self.retrieve_dense(query_embedding, top_k=top_k * 2)

        if not dense_results:
            # Only sparse results available
            return sparse_results[:top_k]

        if not sparse_results:
            # Only dense results available
            return dense_results[:top_k]

        # Combine using RRF or weighted fusion
        if use_rrf:
            combined = self._reciprocal_rank_fusion(
                sparse_results, dense_results, k=rrf_k
            )
        else:
            combined = self._weighted_fusion(
                sparse_results, dense_results, alpha=alpha
            )

        # Sort by score and return top k
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:top_k]

    def _reciprocal_rank_fusion(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) - rank-based fusion.

        RRF formula: score = sum(1 / (k + rank))

        Advantage: No need to normalize scores from different methods.

        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            k: Constant (typically 60)

        Returns:
            Fused results
        """
        # Create rank maps
        sparse_ranks = {r.chunk_id: i + 1 for i, r in enumerate(sparse_results)}
        dense_ranks = {r.chunk_id: i + 1 for i, r in enumerate(dense_results)}

        # Collect all unique chunks
        all_chunk_ids = set(sparse_ranks.keys()) | set(dense_ranks.keys())

        # Compute RRF scores
        rrf_scores = {}
        result_map = {}

        for chunk_id in all_chunk_ids:
            score = 0.0

            if chunk_id in sparse_ranks:
                score += 1.0 / (k + sparse_ranks[chunk_id])
                result_map[chunk_id] = [r for r in sparse_results if r.chunk_id == chunk_id][0]

            if chunk_id in dense_ranks:
                score += 1.0 / (k + dense_ranks[chunk_id])
                if chunk_id not in result_map:
                    result_map[chunk_id] = [r for r in dense_results if r.chunk_id == chunk_id][0]

            rrf_scores[chunk_id] = score

        # Create final results with RRF scores
        results = []
        for chunk_id, score in rrf_scores.items():
            result = result_map[chunk_id]
            result.score = score
            result.retrieval_method = "hybrid"
            results.append(result)

        return results

    def _weighted_fusion(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Weighted score fusion.

        Combined score = alpha * dense_score + (1 - alpha) * sparse_score

        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            alpha: Weight for dense scores (0-1)

        Returns:
            Fused results
        """
        # Normalize sparse scores
        if sparse_results:
            max_sparse = max(r.score for r in sparse_results)
            if max_sparse > 0:
                for r in sparse_results:
                    r.sparse_score = r.score / max_sparse

        # Normalize dense scores
        if dense_results:
            max_dense = max(r.score for r in dense_results)
            if max_dense > 0:
                for r in dense_results:
                    r.dense_score = r.score / max_dense

        # Create combined score map
        result_map = {}

        for result in sparse_results:
            result_map[result.chunk_id] = result
            result.score = (1 - alpha) * (result.sparse_score or 0)

        for result in dense_results:
            if result.chunk_id in result_map:
                # Already have sparse score, add dense
                result_map[result.chunk_id].score += alpha * (result.dense_score or 0)
                result_map[result.chunk_id].dense_score = result.dense_score
                result_map[result.chunk_id].retrieval_method = "hybrid"
            else:
                # Only dense score
                result_map[result.chunk_id] = result
                result.score = alpha * (result.dense_score or 0)

        return list(result_map.values())

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        alpha: float = 0.5,
        use_rrf: bool = True
    ) -> List[RetrievalResult]:
        """
        Main retrieval method - convenience wrapper for retrieve_hybrid.

        Args:
            query: Search query
            query_embedding: Optional query embedding
            top_k: Number of results to return
            alpha: Balance between dense (1.0) and sparse (0.0)
            use_rrf: Use Reciprocal Rank Fusion (recommended)

        Returns:
            List of retrieval results
        """
        return self.retrieve_hybrid(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            alpha=alpha,
            use_rrf=use_rrf
        )
