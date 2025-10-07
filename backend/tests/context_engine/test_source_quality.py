"""
Tests for Source Tracker and Quality Metrics

Combined test file for source attribution and quality measurement.
"""

import pytest
from datetime import datetime, timedelta
from app.context_engine.source_tracker import (
    SourceTracker,
    AuthorityLevel,
    SourceInfo,
    SourceAttribution
)
from app.context_engine.quality_metrics import (
    ContextQualityMetrics,
    ContextQuality
)


# ==================== Source Tracker Tests ====================

def test_register_source():
    """Test source registration"""
    tracker = SourceTracker()

    source = tracker.register_source(
        source_id="doc1",
        source_type="document",
        authority_level=AuthorityLevel.OFFICIAL,
        source_url="https://docs.example.com",
        metadata={"title": "Official Documentation"}
    )

    assert source.source_id == "doc1"
    assert source.authority_level == AuthorityLevel.OFFICIAL
    assert "doc1" in tracker.sources

    print("\n✅ Source registration works")


def test_create_attribution():
    """Test creating source attribution"""
    tracker = SourceTracker()

    tracker.register_source(
        source_id="doc1",
        source_type="document",
        authority_level=AuthorityLevel.VERIFIED
    )

    attribution = tracker.create_attribution(
        source_id="doc1",
        confidence=0.9,
        section="Chapter 3"
    )

    assert attribution.source_id == "doc1"
    assert attribution.confidence_score == 0.9
    assert attribution.section == "Chapter 3"

    print("\n✅ Attribution creation works")


def test_freshness_calculation():
    """Test freshness score calculation"""
    tracker = SourceTracker()

    # Old attribution
    old_time = datetime.now() - timedelta(days=60)
    old_attr = SourceAttribution(
        source_id="old",
        source_type="document",
        source_url=None,
        authority_level=AuthorityLevel.COMMUNITY,
        confidence_score=1.0,
        timestamp=old_time
    )

    # Recent attribution
    recent_time = datetime.now() - timedelta(days=1)
    recent_attr = SourceAttribution(
        source_id="recent",
        source_type="document",
        source_url=None,
        authority_level=AuthorityLevel.OFFICIAL,
        confidence_score=1.0,
        timestamp=recent_time
    )

    old_freshness = tracker.calculate_freshness(old_attr)
    recent_freshness = tracker.calculate_freshness(recent_attr)

    assert recent_freshness > old_freshness
    assert 0.0 <= old_freshness <= 1.0
    assert 0.0 <= recent_freshness <= 1.0

    print(f"\n✅ Freshness calculation works:")
    print(f"   Recent (1 day old): {recent_freshness:.3f}")
    print(f"   Old (60 days old): {old_freshness:.3f}")


def test_rank_attributions():
    """Test attribution ranking"""
    tracker = SourceTracker()

    # Register sources
    tracker.register_source("official", "document", AuthorityLevel.OFFICIAL)
    tracker.register_source("community", "forum", AuthorityLevel.COMMUNITY)

    attr1 = tracker.create_attribution("official", confidence=0.9)
    attr2 = tracker.create_attribution("community", confidence=0.7)

    ranked = tracker.rank_attributions([attr1, attr2])

    assert len(ranked) == 2
    # Official source should rank higher
    assert ranked[0][0].authority_level == AuthorityLevel.OFFICIAL

    print("\n✅ Attribution ranking works")


def test_link_content_to_source():
    """Test linking content to sources"""
    tracker = SourceTracker()

    tracker.register_source("doc1", "document", AuthorityLevel.OFFICIAL)
    attr = tracker.create_attribution("doc1")

    tracker.link_content_to_source("chunk1", attr)

    sources = tracker.get_content_sources("chunk1")
    assert len(sources) == 1
    assert sources[0].source_id == "doc1"

    print("\n✅ Content-source linking works")


# ==================== Quality Metrics Tests ====================

def test_relevance_measurement():
    """Test relevance measurement"""
    metrics = ContextQualityMetrics()

    query = "python programming tutorial"
    context = [
        "Python is a programming language used for development",
        "JavaScript is another programming language"
    ]

    score = metrics.measure_relevance(query, context)

    assert 0.0 <= score <= 1.0
    assert score > 0  # Should have some relevance

    print(f"\n✅ Relevance measurement: {score:.3f}")


def test_completeness_measurement():
    """Test completeness measurement"""
    metrics = ContextQualityMetrics()

    query = "how to install python"
    complete_context = ["how to install python on your system step by step"]
    incomplete_context = ["python is a programming language"]

    complete_score = metrics.measure_completeness(query, complete_context)
    incomplete_score = metrics.measure_completeness(query, incomplete_context)

    assert complete_score > incomplete_score

    print(f"\n✅ Completeness measurement:")
    print(f"   Complete: {complete_score:.3f}")
    print(f"   Incomplete: {incomplete_score:.3f}")


def test_coherence_measurement():
    """Test coherence measurement"""
    metrics = ContextQualityMetrics()

    coherent = [
        "Python is a programming language",
        "Python is used for web development and data science"
    ]

    incoherent = [
        "Python is a programming language",
        "The weather is sunny today"
    ]

    coherent_score = metrics.measure_coherence(coherent)
    incoherent_score = metrics.measure_coherence(incoherent)

    assert coherent_score > incoherent_score

    print(f"\n✅ Coherence measurement:")
    print(f"   Coherent: {coherent_score:.3f}")
    print(f"   Incoherent: {incoherent_score:.3f}")


def test_actionability_measurement():
    """Test actionability measurement"""
    metrics = ContextQualityMetrics()

    actionable = [
        "To install Python, follow these steps: 1) Download from python.org 2) Run installer"
    ]

    vague = [
        "Python is a popular programming language used worldwide"
    ]

    actionable_score = metrics.measure_actionability("how to install python", actionable)
    vague_score = metrics.measure_actionability("how to install python", vague)

    assert actionable_score > vague_score

    print(f"\n✅ Actionability measurement:")
    print(f"   Actionable: {actionable_score:.3f}")
    print(f"   Vague: {vague_score:.3f}")


def test_overall_quality():
    """Test overall quality measurement"""
    metrics = ContextQualityMetrics()

    query = "how to use Python for data science"
    context = [
        "Python is widely used for data science with libraries like pandas and numpy",
        "You can use Python for data analysis by importing these libraries"
    ]

    metadata = [
        {"timestamp": datetime.now(), "authority": "official"},
        {"timestamp": datetime.now() - timedelta(days=10), "authority": "verified"}
    ]

    quality = metrics.measure_overall_quality(query, context, metadata)

    assert isinstance(quality, ContextQuality)
    assert 0.0 <= quality.overall_score <= 1.0
    assert 0.0 <= quality.relevance_score <= 1.0
    assert 0.0 <= quality.completeness_score <= 1.0

    print(f"\n✅ Overall quality measurement:")
    print(f"   Overall: {quality.overall_score:.3f}")
    print(f"   Relevance: {quality.relevance_score:.3f}")
    print(f"   Completeness: {quality.completeness_score:.3f}")
    print(f"   Freshness: {quality.freshness_score:.3f}")
    print(f"   Authority: {quality.authority_score:.3f}")
    print(f"   Coherence: {quality.coherence_score:.3f}")
    print(f"   Actionability: {quality.actionability_score:.3f}")


if __name__ == "__main__":
    # Source Tracker tests
    test_register_source()
    test_create_attribution()
    test_freshness_calculation()
    test_rank_attributions()
    test_link_content_to_source()

    # Quality Metrics tests
    test_relevance_measurement()
    test_completeness_measurement()
    test_coherence_measurement()
    test_actionability_measurement()
    test_overall_quality()

    print("\n" + "="*60)
    print("✅ All source tracker and quality metrics tests passed!")
    print("="*60)
