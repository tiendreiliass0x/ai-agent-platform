from pathlib import Path
import sys
import types
import pytest
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def stub_sentence_transformers(monkeypatch):
    if 'sentence_transformers' not in sys.modules:
        st_stub = types.ModuleType('sentence_transformers')

        class _DummySentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, texts):
                return [[0.0] * 4 for _ in texts]

        class _DummyCrossEncoder:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, pairs):
                return [0.7 for _ in pairs]

        st_stub.SentenceTransformer = _DummySentenceTransformer
        st_stub.CrossEncoder = _DummyCrossEncoder
        sys.modules['sentence_transformers'] = st_stub

    if 'huggingface_hub' not in sys.modules:
        hub_stub = types.ModuleType('huggingface_hub')
        hub_stub.HfApi = object
        hub_stub.HfFolder = object
        hub_stub.Repository = object
        hub_stub.hf_hub_url = lambda *args, **kwargs: ""
        hub_stub.cached_download = lambda *args, **kwargs: ""
        sys.modules['huggingface_hub'] = hub_stub


@pytest.mark.asyncio
async def test_hybrid_retrieve_filters_by_knowledge_pack(monkeypatch):
    from app.services.domain_expertise_service import DomainExpertiseService, RetrievalCandidate
    from app.models.agent import Agent

    service = DomainExpertiseService()

    class AgentRecord:
        def __init__(self, agent_id):
            self.id = agent_id

    async def fake_get_organization_agents(org_id):
        return [AgentRecord(1), AgentRecord(2)]

    async def fake_search(query, agent_id, top_k):
        return [
            {"text": f"content {agent_id}-1", "score": 0.9, "metadata": {"document_id": 11}},
            {"text": f"content {agent_id}-2", "score": 0.4, "metadata": {"document_id": 22}},
        ]

    monkeypatch.setattr(service, "document_processor", types.SimpleNamespace(search_similar_content=fake_search))
    # Patch module-level db_service used inside service
    monkeypatch.setattr(
        "app.services.domain_expertise_service.db_service",
        types.SimpleNamespace(get_organization_agents=fake_get_organization_agents),
    )

    knowledge_pack = types.SimpleNamespace(sources=[types.SimpleNamespace(source_id=11, is_active=True)])

    candidates = await service._hybrid_retrieve(
        message="hello",
        organization_id=9,
        knowledge_pack=knowledge_pack,
        limit=10
    )

    assert all(c.doc_id == "11" for c in candidates)


@pytest.mark.asyncio
async def test_rerank_candidates_uses_recency(monkeypatch):
    from app.services.domain_expertise_service import DomainExpertiseService, RetrievalCandidate

    service = DomainExpertiseService()

    async def fake_rerank(query, items, top_k=None):
        reordered = list(items)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        return reordered

    monkeypatch.setattr("app.services.domain_expertise_service.reranker_service", types.SimpleNamespace(rerank=fake_rerank))

    recent = RetrievalCandidate(
        doc_id="a",
        content="recent",
        score=0.5,
        timestamp=datetime.utcnow() - timedelta(days=1)
    )
    old = RetrievalCandidate(
        doc_id="b",
        content="old",
        score=0.6,
        timestamp=datetime.utcnow() - timedelta(days=40)
    )

    ranked = await service._rerank_candidates([recent, old], "question", None)
    assert ranked[0].doc_id == "b"
    assert ranked[1].doc_id == "a"
    # Recent item got a boost relative to its base
    assert ranked[1].score > 0.5


@pytest.mark.asyncio
async def test_answer_with_domain_expertise_triggers_web_search(monkeypatch):
    from app.services.domain_expertise_service import DomainExpertiseService, GroundedResponse

    service = DomainExpertiseService()

    async def fake_hybrid(*args, **kwargs):
        return []

    async def fake_web_search(*args, **kwargs):
        from app.services.domain_expertise_service import RetrievalCandidate
        return [
            RetrievalCandidate(
                doc_id="web_https://example.com",
                content="snippet",
                score=0.8,
                source_url="https://example.com",
                doc_title="Example",
                source_type="web",
            )
        ]

    async def fake_plan(*args, **kwargs):
        return {"plan": "respond"}

    async def fake_synthesize(*args, **kwargs):
        return GroundedResponse(
            answer="final",
            confidence_score=0.7,
            sources=[{"source": "web"}],
            grounding_mode="strict",
            persona_applied="persona",
            web_search_used=True
        )

    service._hybrid_retrieve = fake_hybrid
    monkeypatch.setattr(service, "_web_search", fake_web_search)
    monkeypatch.setattr(service, "_plan_answer", fake_plan)
    monkeypatch.setattr(service, "_synthesize_response", fake_synthesize)
    # Force path: fresh info detected so web search runs, but enough support after merge
    monkeypatch.setattr(service, "_needs_fresh_info", lambda *args, **kwargs: True)
    monkeypatch.setattr(service, "_has_sufficient_support", lambda *args, **kwargs: True)

    agent = types.SimpleNamespace(
        domain_expertise_enabled=True,
        tool_policy={"web_search": True, "site_search": []},
        grounding_mode="strict",
        persona_id=1,
        knowledge_pack_id=None,
    )
    persona = types.SimpleNamespace(template_name="sales_rep")
    async def _fake_load_persona(*args, **kwargs):
        return persona
    monkeypatch.setattr(service, "_load_persona", _fake_load_persona)
    async def _fake_load_pack(*args, **kwargs):
        return None
    monkeypatch.setattr(service, "_load_knowledge_pack", _fake_load_pack)

    response = await service.answer_with_domain_expertise("Need latest info", agent, types.SimpleNamespace(id=1))
    assert response.web_search_used is True
    assert response.sources[0]["source"] == "web"
