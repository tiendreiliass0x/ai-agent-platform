"""
Source Tracker - Provenance & Attribution

Tracks the origin of every piece of information with:
- Source document/URL
- Confidence scores
- Timestamp (when was this information current)
- Authority level (official docs vs forum posts)

Enables:
- Citing sources in responses
- Prioritizing trusted sources
- Detecting contradictions between sources
- Tracking information freshness
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


class AuthorityLevel(str, Enum):
    """Authority level of information sources"""
    OFFICIAL = "official"  # Official documentation, company sources
    VERIFIED = "verified"  # Verified third-party sources
    COMMUNITY = "community"  # Community forums, discussions
    UNVERIFIED = "unverified"  # Unknown or untrusted sources


@dataclass
class SourceInfo:
    """Information about a registered source"""
    source_id: str
    source_type: str
    authority_level: AuthorityLevel
    source_url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    title: Optional[str] = None
    author: Optional[str] = None


@dataclass
class SourceAttribution:
    """Attribution information for a piece of content"""
    source_id: str
    source_type: str
    source_url: Optional[str]
    authority_level: AuthorityLevel
    confidence_score: float  # 0-1
    timestamp: datetime
    section: Optional[str] = None  # Which section/page of source
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        """Generate human-readable citation"""
        parts = []

        if self.metadata.get("title"):
            parts.append(f"\"{self.metadata['title']}\"")

        if self.source_url:
            parts.append(f"({self.source_url})")
        elif self.source_id:
            parts.append(f"(Source: {self.source_id})")

        if self.section:
            parts.append(f"Section: {self.section}")

        return " ".join(parts)


class SourceTracker:
    """
    Track provenance and attribution for all knowledge.

    Features:
    - Source registration with authority levels
    - Attribution creation and linking
    - Confidence scoring
    - Freshness calculation
    - Basic contradiction detection
    """

    def __init__(self):
        """Initialize source tracker"""
        self.sources: Dict[str, SourceInfo] = {}
        self.content_sources: Dict[str, List[SourceAttribution]] = defaultdict(list)

    def register_source(
        self,
        source_id: str,
        source_type: str,
        authority_level: AuthorityLevel,
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SourceInfo:
        """
        Register a new source in the tracking system.

        Args:
            source_id: Unique identifier
            source_type: Type of source (document, webpage, api_response, etc.)
            authority_level: Authority level
            source_url: Optional URL
            metadata: Additional metadata (title, author, etc.)

        Returns:
            SourceInfo object
        """
        source = SourceInfo(
            source_id=source_id,
            source_type=source_type,
            authority_level=authority_level,
            source_url=source_url,
            timestamp=datetime.now(),
            metadata=metadata or {},
            title=metadata.get("title") if metadata else None,
            author=metadata.get("author") if metadata else None
        )

        self.sources[source_id] = source
        return source

    def create_attribution(
        self,
        source_id: str,
        confidence: float = 1.0,
        section: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SourceAttribution:
        """
        Create attribution for content.

        Args:
            source_id: Source identifier
            confidence: Confidence score (0-1)
            section: Section within source
            metadata: Additional metadata

        Returns:
            SourceAttribution object
        """
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} not registered")

        source = self.sources[source_id]

        attribution = SourceAttribution(
            source_id=source_id,
            source_type=source.source_type,
            source_url=source.source_url,
            authority_level=source.authority_level,
            confidence_score=max(0.0, min(1.0, confidence)),
            timestamp=source.timestamp,
            section=section,
            metadata=metadata or {}
        )

        return attribution

    def link_content_to_source(
        self,
        content_id: str,
        attribution: SourceAttribution
    ) -> None:
        """
        Link content to its source attribution.

        Args:
            content_id: Unique content identifier (e.g., chunk_id)
            attribution: Source attribution
        """
        self.content_sources[content_id].append(attribution)

    def get_content_sources(self, content_id: str) -> List[SourceAttribution]:
        """
        Get all source attributions for a piece of content.

        Args:
            content_id: Content identifier

        Returns:
            List of source attributions
        """
        return self.content_sources.get(content_id, [])

    def calculate_freshness(
        self,
        attribution: SourceAttribution,
        decay_half_life: timedelta = timedelta(days=30)
    ) -> float:
        """
        Calculate freshness score based on timestamp.

        Uses exponential decay: freshness = 0.5 ^ (age / half_life)

        Args:
            attribution: Source attribution
            decay_half_life: Time period for 50% freshness decay

        Returns:
            Freshness score (0-1)
        """
        age = datetime.now() - attribution.timestamp
        age_days = age.total_seconds() / 86400
        half_life_days = decay_half_life.total_seconds() / 86400

        if age_days <= 0:
            return 1.0

        # Exponential decay
        freshness = 0.5 ** (age_days / half_life_days)
        return max(0.0, min(1.0, freshness))

    def get_authority_score(self, authority_level: AuthorityLevel) -> float:
        """
        Convert authority level to numeric score.

        Args:
            authority_level: Authority level

        Returns:
            Authority score (0-1)
        """
        scores = {
            AuthorityLevel.OFFICIAL: 1.0,
            AuthorityLevel.VERIFIED: 0.8,
            AuthorityLevel.COMMUNITY: 0.5,
            AuthorityLevel.UNVERIFIED: 0.3
        }
        return scores.get(authority_level, 0.3)

    def rank_attributions(
        self,
        attributions: List[SourceAttribution],
        freshness_weight: float = 0.3,
        authority_weight: float = 0.4,
        confidence_weight: float = 0.3
    ) -> List[Tuple[SourceAttribution, float]]:
        """
        Rank source attributions by quality.

        Args:
            attributions: List of attributions to rank
            freshness_weight: Weight for freshness score
            authority_weight: Weight for authority score
            confidence_weight: Weight for confidence score

        Returns:
            List of (attribution, combined_score) tuples, sorted by score desc
        """
        ranked = []

        for attr in attributions:
            freshness = self.calculate_freshness(attr)
            authority = self.get_authority_score(attr.authority_level)
            confidence = attr.confidence_score

            # Weighted combination
            combined_score = (
                freshness_weight * freshness +
                authority_weight * authority +
                confidence_weight * confidence
            )

            ranked.append((attr, combined_score))

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def detect_contradictions(
        self,
        statements: List[Tuple[str, SourceAttribution]],
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detect potential contradictions between sources.

        Simple heuristic: statements from different sources about same topic
        with low similarity might contradict.

        Args:
            statements: List of (statement_text, attribution) tuples
            similarity_threshold: Threshold for contradiction detection

        Returns:
            List of detected contradictions with evidence
        """
        contradictions = []

        # Group statements by source
        source_statements = defaultdict(list)
        for text, attr in statements:
            source_statements[attr.source_id].append((text, attr))

        # Check for different sources making different claims
        if len(source_statements) > 1:
            # Simple contradiction detection:
            # If we have multiple sources, flag it as potential contradiction
            sources = list(source_statements.keys())

            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    source1 = sources[i]
                    source2 = sources[j]

                    statements1 = source_statements[source1]
                    statements2 = source_statements[source2]

                    # Flag as potential contradiction
                    contradictions.append({
                        "type": "multiple_sources",
                        "source1": source1,
                        "source2": source2,
                        "statements1": [s[0] for s in statements1],
                        "statements2": [s[0] for s in statements2],
                        "confidence": 0.5  # Low confidence without semantic analysis
                    })

        return contradictions

    def get_best_attribution(
        self,
        content_id: str
    ) -> Optional[SourceAttribution]:
        """
        Get highest quality attribution for content.

        Args:
            content_id: Content identifier

        Returns:
            Best attribution or None
        """
        attributions = self.get_content_sources(content_id)

        if not attributions:
            return None

        ranked = self.rank_attributions(attributions)
        return ranked[0][0] if ranked else None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Summary dict with counts and statistics
        """
        authority_counts = defaultdict(int)
        for source in self.sources.values():
            authority_counts[source.authority_level.value] += 1

        return {
            "total_sources": len(self.sources),
            "total_content_links": len(self.content_sources),
            "authority_distribution": dict(authority_counts),
            "source_types": list(set(s.source_type for s in self.sources.values()))
        }
