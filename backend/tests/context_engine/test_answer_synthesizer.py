"""
Tests for AnswerSynthesizer
"""

from app.context_engine.answer_synthesizer import AnswerSynthesizer


def test_synthesize_with_contexts():
    synthesizer = AnswerSynthesizer()
    contexts = [
        {"text": "Pinecone offers a managed vector database service.", "source_id": "doc_pinecone", "section": "Overview"},
        {"text": "It integrates with OpenAI embeddings for semantic search.", "source_id": "doc_integration"},
    ]

    answer = synthesizer.synthesize("How does Pinecone integrate?", contexts, confidence=0.82)

    assert "Pinecone" in answer.summary
    assert len(answer.supporting_points) == 2
    assert answer.citations == ["doc_pinecone (Overview)", "doc_integration"]
    assert answer.confidence == 0.82


def test_synthesize_without_contexts():
    synthesizer = AnswerSynthesizer()
    answer = synthesizer.synthesize("Explain the feature.", [])

    assert "could not find" in answer.summary.lower()
    assert answer.supporting_points == []
