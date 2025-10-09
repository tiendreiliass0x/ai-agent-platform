"""
Tests for Query Understanding Engine
"""

import json
import pytest

from app.context_engine.query_understanding import QueryUnderstandingEngine


def test_intent_detection_simple():
    engine = QueryUnderstandingEngine()
    result = engine.analyze("What is the difference between SaaS and PaaS?")

    assert result.question_type == "comparison"
    assert "comparison" in result.intents
    assert result.complexity == "complex"
    assert "SaaS" in result.entities
    assert "PaaS" in result.entities
    assert len(result.sub_queries) >= 1


def test_expansion_terms_and_history_entity_resolution():
    engine = QueryUnderstandingEngine()
    history = [
        "We were discussing Pinecone vector database and Redis.",
        "It provides managed vector storage.",
    ]
    result = engine.analyze("How does it handle pricing tiers?", conversation_history=history)

    # Pricing should trigger expansion hints
    assert any(term in result.expansion_terms for term in ["tiers", "plans", "discount"])
    # Should fallback to history entities for "it"
    assert "Pinecone vector database" in result.entities or "Pinecone" in result.entities


def test_query_decomposition():
    engine = QueryUnderstandingEngine()
    query = "Explain the onboarding steps and how to integrate with Slack."
    result = engine.analyze(query)

    assert result.complexity == "complex"
    assert "procedural" in result.intents
    assert len(result.sub_queries) == 2


class _FakeLLM:
    def __init__(self, response: str, should_fail: bool = False):
        self.response = response
        self.should_fail = should_fail
        self.calls = 0

    async def generate_response(self, *_, **__):
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("LLM failure")
        return self.response


@pytest.mark.asyncio
async def test_llm_analysis_overrides_rule_based():
    payload = json.dumps({
        "primary_intent": "comparison",
        "sub_intents": ["procedural"],
        "entities": ["Plan A", "Plan B"],
        "sub_questions": ["What features does Plan A offer?"],
        "query_reformulations": ["Compare Plan A and Plan B for enterprises"],
        "implicit_requirements": ["Focus on security"],
        "complexity": "complex"
    })

    engine = QueryUnderstandingEngine(
        use_llm=True,
        llm_service=_FakeLLM(payload),
        llm_temperature=0.0,
    )

    history = ["We covered Plan A yesterday.", "Plan B is for enterprise customers."]
    result = await engine.analyze_async("How do they differ?", conversation_history=history)

    assert result.question_type == "comparison"
    assert "Plan A" in result.entities
    assert "Compare Plan A and Plan B for enterprises" in result.expansion_terms
    assert result.metadata.get("intent_source") == "llm"
    assert result.complexity == "complex"


@pytest.mark.asyncio
async def test_llm_analysis_graceful_fallback_on_error():
    engine = QueryUnderstandingEngine(
        use_llm=True,
        llm_service=_FakeLLM("{}", should_fail=True),
    )

    result = await engine.analyze_async("How to install Python?")
    assert result.question_type == "procedural"
    assert "procedural" in result.intents
    # Ensure fallback metadata doesn't falsely mark LLM
    assert result.metadata.get("intent_source") != "llm"
