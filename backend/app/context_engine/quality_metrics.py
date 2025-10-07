"""
Context Quality Metrics - Measurable Excellence

Quantifies the quality of retrieved context across multiple dimensions:
- Relevance: How well does it match the query?
- Completeness: Does it have all necessary information?
- Freshness: How recent is the information?
- Authority: How trustworthiness is the source?
- Coherence: How well do the pieces fit together?
- Actionability: Can the user act on this information?
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ContextQuality:
    """Quality metrics for retrieved context"""
    relevance_score: float  # 0-1: Semantic match to query
    completeness_score: float  # 0-1: Information sufficiency
    freshness_score: float  # 0-1: Recency of information
    authority_score: float  # 0-1: Source trustworthiness
    coherence_score: float  # 0-1: Internal consistency
    actionability_score: float  # 0-1: Practical usefulness

    overall_score: float = 0.0  # Weighted combination

    def __post_init__(self):
        """Calculate overall score if not provided"""
        if self.overall_score == 0.0:
            self.overall_score = self._calculate_overall()

    def _calculate_overall(self) -> float:
        """
        Calculate weighted overall quality score.

        Weights:
            - Relevance: 25%
            - Completeness: 20%
            - Freshness: 15%
            - Authority: 15%
            - Coherence: 15%
            - Actionability: 10%
        """
        return (
            self.relevance_score * 0.25 +
            self.completeness_score * 0.20 +
            self.freshness_score * 0.15 +
            self.authority_score * 0.15 +
            self.coherence_score * 0.15 +
            self.actionability_score * 0.10
        )


class ContextQualityMetrics:
    """
    Measure and track context quality across all dimensions.

    Uses heuristics and simple metrics for quality assessment.
    """

    def __init__(self):
        """Initialize quality metrics system"""
        pass

    def measure_relevance(
        self,
        query: str,
        retrieved_context: List[str]
    ) -> float:
        """
        Measure how relevant the retrieved context is to the query.

        Uses keyword overlap as simple heuristic.

        Args:
            query: User query
            retrieved_context: List of retrieved text chunks

        Returns:
            Relevance score (0-1)
        """
        if not retrieved_context:
            return 0.0

        # Simple keyword-based relevance
        query_words = set(query.lower().split())

        relevance_scores = []
        for context in retrieved_context:
            context_words = set(context.lower().split())
            if not context_words:
                relevance_scores.append(0.0)
                continue

            # Jaccard similarity
            intersection = query_words & context_words
            union = query_words | context_words

            if union:
                relevance_scores.append(len(intersection) / len(union))
            else:
                relevance_scores.append(0.0)

        # Average relevance
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    def measure_completeness(
        self,
        query: str,
        retrieved_context: List[str],
        required_aspects: Optional[List[str]] = None
    ) -> float:
        """
        Measure if the context has all necessary information.

        Checks if key query terms and required aspects are covered.

        Args:
            query: User query
            retrieved_context: Retrieved text chunks
            required_aspects: Optional list of required information aspects

        Returns:
            Completeness score (0-1)
        """
        if not retrieved_context:
            return 0.0

        all_context = " ".join(retrieved_context).lower()

        # Check query terms coverage
        query_terms = set(query.lower().split())
        covered_terms = sum(1 for term in query_terms if term in all_context)
        term_coverage = covered_terms / len(query_terms) if query_terms else 0.0

        # Check required aspects if provided
        if required_aspects:
            covered_aspects = sum(1 for aspect in required_aspects if aspect.lower() in all_context)
            aspect_coverage = covered_aspects / len(required_aspects)

            # Combine both
            return (term_coverage + aspect_coverage) / 2

        return term_coverage

    def measure_freshness(
        self,
        context_timestamps: List[datetime],
        decay_half_life: timedelta = timedelta(days=30)
    ) -> float:
        """
        Measure how fresh/recent the information is.

        Uses exponential decay: freshness = 0.5 ^ (age / half_life)

        Args:
            context_timestamps: When each piece was created/updated
            decay_half_life: Time period for 50% relevance decay

        Returns:
            Freshness score (0-1)
        """
        if not context_timestamps:
            return 0.5  # Neutral score

        now = datetime.now()
        freshness_scores = []

        for timestamp in context_timestamps:
            age = now - timestamp
            age_days = age.total_seconds() / 86400
            half_life_days = decay_half_life.total_seconds() / 86400

            if age_days <= 0:
                freshness_scores.append(1.0)
            else:
                # Exponential decay
                freshness = 0.5 ** (age_days / half_life_days)
                freshness_scores.append(max(0.0, min(1.0, freshness)))

        return sum(freshness_scores) / len(freshness_scores)

    def measure_authority(
        self,
        source_authorities: List[str]
    ) -> float:
        """
        Measure the authority/trustworthiness of sources.

        Args:
            source_authorities: Authority levels of sources

        Returns:
            Authority score (0-1)
        """
        if not source_authorities:
            return 0.5  # Neutral score

        authority_scores = {
            "official": 1.0,
            "verified": 0.8,
            "community": 0.5,
            "unverified": 0.3
        }

        scores = [authority_scores.get(auth.lower(), 0.5) for auth in source_authorities]
        return sum(scores) / len(scores)

    def measure_coherence(
        self,
        retrieved_context: List[str]
    ) -> float:
        """
        Measure how coherent/consistent the context is.

        Simple heuristic: check vocabulary overlap between chunks.

        Args:
            retrieved_context: Retrieved text chunks

        Returns:
            Coherence score (0-1)
        """
        if len(retrieved_context) <= 1:
            return 1.0  # Single chunk is coherent by definition

        # Measure vocabulary overlap between consecutive chunks
        coherence_scores = []

        for i in range(len(retrieved_context) - 1):
            words1 = set(retrieved_context[i].lower().split())
            words2 = set(retrieved_context[i + 1].lower().split())

            if not words1 or not words2:
                continue

            # Jaccard similarity
            intersection = words1 & words2
            union = words1 | words2

            if union:
                coherence_scores.append(len(intersection) / len(union))

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5

    def measure_actionability(
        self,
        query: str,
        retrieved_context: List[str]
    ) -> float:
        """
        Measure how actionable/practical the information is.

        Looks for action-oriented keywords and specific information.

        Args:
            query: User query
            retrieved_context: Retrieved text chunks

        Returns:
            Actionability score (0-1)
        """
        if not retrieved_context:
            return 0.0

        all_context = " ".join(retrieved_context).lower()

        # Action-oriented keywords
        action_keywords = [
            "how", "steps", "guide", "tutorial", "instructions",
            "example", "code", "command", "api", "method",
            "function", "use", "implement", "create", "configure"
        ]

        # Count action keywords
        keyword_count = sum(1 for keyword in action_keywords if keyword in all_context)

        # Check for specific details (URLs, code blocks, numbers)
        has_urls = "http" in all_context or "www." in all_context
        has_code = "`" in " ".join(retrieved_context) or "```" in " ".join(retrieved_context)
        has_numbers = any(char.isdigit() for char in all_context)

        # Combine signals
        keyword_score = min(1.0, keyword_count / 3)  # 3+ keywords = max score
        specificity_score = sum([has_urls, has_code, has_numbers]) / 3

        return (keyword_score + specificity_score) / 2

    def measure_overall_quality(
        self,
        query: str,
        retrieved_context: List[str],
        context_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> ContextQuality:
        """
        Measure overall context quality across all dimensions.

        Args:
            query: User query
            retrieved_context: Retrieved text chunks
            context_metadata: Metadata for each chunk (timestamps, sources, etc.)

        Returns:
            ContextQuality object with all scores
        """
        context_metadata = context_metadata or []

        # Extract metadata
        timestamps = []
        authorities = []

        for metadata in context_metadata:
            if "timestamp" in metadata:
                timestamps.append(metadata["timestamp"])
            if "authority" in metadata:
                authorities.append(metadata["authority"])

        # Measure all dimensions
        relevance = self.measure_relevance(query, retrieved_context)
        completeness = self.measure_completeness(query, retrieved_context)
        freshness = self.measure_freshness(timestamps) if timestamps else 0.5
        authority = self.measure_authority(authorities) if authorities else 0.5
        coherence = self.measure_coherence(retrieved_context)
        actionability = self.measure_actionability(query, retrieved_context)

        return ContextQuality(
            relevance_score=relevance,
            completeness_score=completeness,
            freshness_score=freshness,
            authority_score=authority,
            coherence_score=coherence,
            actionability_score=actionability
        )
