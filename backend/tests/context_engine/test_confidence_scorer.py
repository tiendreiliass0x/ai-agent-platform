"""
Tests for ConfidenceScorer
"""

from app.context_engine.confidence_scorer import ConfidenceScorer


def test_confidence_scoring_with_high_signals():
    scorer = ConfidenceScorer(escalation_threshold=0.4)
    breakdown = scorer.score(
        coverage=0.9,
        agreement=0.8,
        authority=0.85,
        freshness=0.7,
        retrieval_signal=0.9,
    )

    assert breakdown.overall > 0.75
    assert breakdown.notes is None


def test_confidence_scoring_below_threshold_triggers_note():
    scorer = ConfidenceScorer(escalation_threshold=0.5)
    breakdown = scorer.score(
        coverage=0.2,
        agreement=0.3,
        authority=0.4,
        freshness=0.2,
        retrieval_signal=0.3,
        metadata={"escalation_message": "Ask user for clarification."},
    )

    assert breakdown.overall < 0.5
    assert breakdown.notes == "Ask user for clarification."


def test_aggregate_retrieval_scores_normalization():
    scores = [12.0, 5.0, 2.0]
    aggregated = ConfidenceScorer.aggregate_retrieval_scores(scores)
    assert 0 < aggregated <= 1
