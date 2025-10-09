"""
Confidence Scoring - Assess Answer Reliability

Calculates confidence for candidate answers using multiple signals:
    - Evidence coverage and agreement
    - Source authority and freshness
    - Retrieval and rerank scores
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ConfidenceBreakdown:
    overall: float
    coverage: float
    agreement: float
    authority: float
    freshness: float
    retrieval: float
    notes: Optional[str] = None


class ConfidenceScorer:
    """Combine heuristics to produce a confidence score."""

    def __init__(
        self,
        coverage_weight: float = 0.25,
        agreement_weight: float = 0.2,
        authority_weight: float = 0.2,
        freshness_weight: float = 0.15,
        retrieval_weight: float = 0.2,
        escalation_threshold: float = 0.35,
    ) -> None:
        self.coverage_weight = coverage_weight
        self.agreement_weight = agreement_weight
        self.authority_weight = authority_weight
        self.freshness_weight = freshness_weight
        self.retrieval_weight = retrieval_weight
        self.escalation_threshold = escalation_threshold

    def score(
        self,
        coverage: float,
        agreement: float,
        authority: float,
        freshness: float,
        retrieval_signal: float,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ConfidenceBreakdown:
        metadata = metadata or {}

        coverage = self._clamp(coverage)
        agreement = self._clamp(agreement)
        authority = self._clamp(authority)
        freshness = self._clamp(freshness)
        retrieval_signal = self._clamp(retrieval_signal)

        overall = (
            coverage * self.coverage_weight
            + agreement * self.agreement_weight
            + authority * self.authority_weight
            + freshness * self.freshness_weight
            + retrieval_signal * self.retrieval_weight
        )

        notes = None
        if overall < self.escalation_threshold:
            notes = metadata.get(
                "escalation_message",
                "Confidence below threshold; escalate to human or request clarification.",
            )

        return ConfidenceBreakdown(
            overall=overall,
            coverage=coverage,
            agreement=agreement,
            authority=authority,
            freshness=freshness,
            retrieval=retrieval_signal,
            notes=notes,
        )

    @staticmethod
    def aggregate_retrieval_scores(scores: List[float]) -> float:
        if not scores:
            return 0.0
        top_scores = sorted(scores, reverse=True)[:3]
        normalized = [ConfidenceScorer._normalize_score(score) for score in top_scores]
        return sum(normalized) / len(normalized)

    @staticmethod
    def _normalize_score(score: float) -> float:
        # BM25 scores or cosine similarities can exceed 1; squash to 0-1.
        if score <= 0:
            return 0.0
        return min(score / (score + 1), 1.0)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
