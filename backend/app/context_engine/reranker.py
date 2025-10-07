"""
Reranker - Context Quality Refinement

Reranks retrieved chunks using cross-encoder models for improved relevance.

Why Reranking?
- Initial retrieval (BM25/vector) is fast but may miss semantic nuances
- Cross-encoders jointly encode query+document for better relevance scoring
- Typical pipeline: Retrieve 20-100 candidates → Rerank top 5-10

Supported Models:
- Jina Reranker v2 (278M params, multilingual)
- Custom cross-encoders
- Fallback to score-based reranking
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class RerankResult:
    """Result from reranking"""
    text: str
    original_score: float
    rerank_score: float
    final_score: float  # Combined score
    rank: int
    metadata: Dict[str, Any]


class Reranker:
    """
    Rerank retrieved chunks for improved relevance.

    Uses cross-encoder models when available, falls back to heuristics.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        use_cross_encoder: bool = True,
        top_k: int = 5,
        score_weight: float = 0.5  # Weight for original vs rerank score
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace model name
            use_cross_encoder: Use cross-encoder model (requires transformers)
            top_k: Number of results to return after reranking
            score_weight: Weight for combining original + rerank scores (0-1)
        """
        self.model_name = model_name
        self.use_cross_encoder = use_cross_encoder
        self.top_k = top_k
        self.score_weight = score_weight
        self._model = None

    @property
    def model(self):
        """Lazy load reranker model"""
        if self._model is None and self.use_cross_encoder:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                print(f"Loading reranker model: {self.model_name}")
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                # Move to GPU if available
                if torch.cuda.is_available():
                    self._model = self._model.cuda()

                self._model.eval()
                print("✓ Reranker model loaded successfully")

            except ImportError:
                print("Warning: transformers not installed. Using fallback reranking.")
                self.use_cross_encoder = False
            except Exception as e:
                print(f"Warning: Failed to load reranker model: {e}")
                self.use_cross_encoder = False

        return self._model

    def rerank_with_cross_encoder(
        self,
        query: str,
        documents: List[str],
        original_scores: Optional[List[float]] = None
    ) -> List[float]:
        """
        Rerank documents using cross-encoder model.

        Args:
            query: Search query
            documents: List of document texts
            original_scores: Optional original retrieval scores

        Returns:
            List of reranking scores
        """
        if self.model is None:
            # Fallback to original scores
            return original_scores if original_scores else [0.0] * len(documents)

        try:
            import torch

            # Prepare pairs
            pairs = [[query, doc] for doc in documents]

            # Tokenize
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )

            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get scores
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)

                # Apply sigmoid for normalized scores
                scores = torch.sigmoid(scores)

            return scores.cpu().numpy().tolist()

        except Exception as e:
            print(f"Error during reranking: {e}")
            return original_scores if original_scores else [0.0] * len(documents)

    def rerank_fallback(
        self,
        query: str,
        documents: List[str],
        original_scores: List[float]
    ) -> List[float]:
        """
        Fallback reranking using simple heuristics.

        Combines:
        - Original retrieval score
        - Document length (prefer moderate length)
        - Query term coverage

        Args:
            query: Search query
            documents: List of documents
            original_scores: Original retrieval scores

        Returns:
            Adjusted scores
        """
        query_terms = set(query.lower().split())
        rerank_scores = []

        for doc, orig_score in zip(documents, original_scores):
            score = orig_score

            # Query term coverage bonus
            doc_words = set(doc.lower().split())
            coverage = len(query_terms & doc_words) / max(len(query_terms), 1)
            score += coverage * 0.2

            # Length normalization (prefer 100-500 words)
            doc_len = len(doc.split())
            if 100 <= doc_len <= 500:
                score += 0.1
            elif doc_len < 50:
                score -= 0.1

            rerank_scores.append(score)

        return rerank_scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        original_scores: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: Retrieved documents
            original_scores: Optional original retrieval scores
            metadata: Optional metadata for each document

        Returns:
            List of RerankResult objects, sorted by final score
        """
        if not documents:
            return []

        # Default scores
        if original_scores is None:
            original_scores = [1.0] * len(documents)

        if metadata is None:
            metadata = [{} for _ in documents]

        # Get rerank scores
        if self.use_cross_encoder and self.model is not None:
            rerank_scores = self.rerank_with_cross_encoder(
                query, documents, original_scores
            )
        else:
            rerank_scores = self.rerank_fallback(
                query, documents, original_scores
            )

        # Combine scores
        results = []
        for i, (doc, orig_score, rerank_score) in enumerate(
            zip(documents, original_scores, rerank_scores)
        ):
            # Weighted combination
            final_score = (
                self.score_weight * orig_score +
                (1 - self.score_weight) * rerank_score
            )

            results.append(RerankResult(
                text=doc,
                original_score=orig_score,
                rerank_score=rerank_score,
                final_score=final_score,
                rank=i,  # Will be updated after sorting
                metadata=metadata[i]
            ))

        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i

        # Return top k
        return results[:self.top_k]

    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        scores_list: Optional[List[List[float]]] = None,
        metadata_list: Optional[List[List[Dict[str, Any]]]] = None
    ) -> List[List[RerankResult]]:
        """
        Rerank multiple query-document sets in batch.

        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            scores_list: Optional list of score lists
            metadata_list: Optional list of metadata lists

        Returns:
            List of reranked results for each query
        """
        results = []

        for i, (query, documents) in enumerate(zip(queries, documents_list)):
            scores = scores_list[i] if scores_list else None
            metadata = metadata_list[i] if metadata_list else None

            query_results = self.rerank(query, documents, scores, metadata)
            results.append(query_results)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dict
        """
        return {
            "model_name": self.model_name,
            "using_cross_encoder": self.use_cross_encoder and self.model is not None,
            "top_k": self.top_k,
            "score_weight": self.score_weight,
            "model_loaded": self._model is not None
        }
