"""
Tests for Multi-Hop Reasoner
"""

from app.context_engine.multi_hop_reasoner import MultiHopReasoner
from app.context_engine.knowledge_graph_builder import KnowledgeGraphBuilder


def build_sample_graph() -> KnowledgeGraphBuilder:
    builder = KnowledgeGraphBuilder()
    documents = [
        "Pinecone provides vector databases. Pinecone integrates with OpenAI.",
        "OpenAI offers embeddings. Embeddings power semantic search.",
    ]
    builder.build_graph(documents)
    return builder


def test_reasoning_plan_with_graph():
    kg = build_sample_graph()
    reasoner = MultiHopReasoner(knowledge_graph=kg)
    chunks = [
        {"chunk_id": "chunk1", "text": "Pinecone integrates with OpenAI.", "metadata": {"entities": ["Pinecone", "OpenAI"]}},
        {"chunk_id": "chunk2", "text": "Embeddings power semantic search.", "metadata": {"entities": ["Embeddings"]}},
    ]

    plan = reasoner.plan("How do Pinecone and OpenAI work together?", chunks)

    assert len(plan.steps) >= 2
    assert plan.steps[0].supporting_chunks[0] == "chunk1"
    assert any("Inspect relationships" in step.description for step in plan.steps)


def test_plan_without_initial_chunks():
    reasoner = MultiHopReasoner()
    plan = reasoner.plan("Explain the relationship between Redis and Celery.", [])

    assert plan.requires_additional_retrieval is True
    assert "Perform initial retrieval" in plan.steps[0].description
