"""
Tests for Contradiction Detector
"""

from app.context_engine.contradiction_detector import ContradictionDetector, Statement


def test_detects_conflict_between_sources():
    detector = ContradictionDetector()
    statements = [
        Statement(
            text="Pinecone supports real-time updates.",
            source_id="doc_official",
            authority="official",
            timestamp="2024-01-01",
        ),
        Statement(
            text="Pinecone does not support real-time updates.",
            source_id="community_post",
            authority="community",
            timestamp="2023-06-01",
        ),
    ]

    findings = detector.detect(statements)
    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert "Conflicting statements" in findings[0].rationale


def test_no_conflict_same_polarity():
    detector = ContradictionDetector()
    statements = [
        Statement(text="Redis caching is enabled.", source_id="doc1", authority="official"),
        Statement(text="Redis caching is enabled for all plans.", source_id="doc2", authority="verified"),
    ]

    findings = detector.detect(statements)
    assert findings == []
