from pathlib import Path
import sys
import types
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


class DummyGemini:
    async def generate_response(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2048):
        return "Gemini response"


class DummyPersonalityService:
    async def get_agent_personality(self, agent_id, db_session):
        return {"voice": "friendly"}

    def inject_personality_into_prompt(self, base_prompt, personality, user_query, context):
        return f"{base_prompt}\nVOICE:{personality['voice']}"

    def enhance_response_with_personality(self, response, personality, user_query, conversation_history):
        return f"{response} ::persona::{personality['voice']}"


@pytest.fixture(autouse=True)
def stub_external_modules(monkeypatch):
    # Stub sentence_transformers
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
                return [0.5 for _ in pairs]

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
async def test_rag_service_generate_response(monkeypatch):
    from app.services import rag_service

    service = rag_service.RAGService()

    class StubDocumentProcessor:
        async def search_similar_content(self, query, agent_id, top_k):
            return [
                {"text": "Chunk A", "score": 0.9, "metadata": {"source": "docA", "chunk_id": "1"}},
                {"text": "Chunk B", "score": 0.5, "metadata": {"source": "docB", "chunk_id": "2"}},
            ]

    class StubReranker:
        async def rerank(self, query, items, top_k=None):
            return list(reversed(items))

    monkeypatch.setattr(rag_service, "document_processor", StubDocumentProcessor(), raising=False)
    service.document_processor = StubDocumentProcessor()
    service.reranker = StubReranker()

    monkeypatch.setattr(rag_service, "personality_service", DummyPersonalityService(), raising=False)
    service.gemini_service = DummyGemini()

    result = await service.generate_response(
        query="How to reset password?",
        agent_id=123,
        conversation_history=[{"role": "user", "content": "Hello"}],
        system_prompt="You are helpful.",
        db_session=object()
    )

    assert result["response"].startswith("Gemini response")
    assert result["response"].endswith("::persona::friendly")
    assert {src["source"] for src in result["sources"]} == {"docA", "docB"}
    assert result["sources"][0]["source"] == "docB"
    assert result["context_used"] == 2


@pytest.mark.asyncio
async def test_rag_service_retrieve_context_return_full(monkeypatch):
    from app.services import rag_service

    service = rag_service.RAGService()

    class StubDocumentProcessor:
        async def search_similar_content(self, query, agent_id, top_k):
            return [
                {"text": "Paragraph 1", "score": 0.8, "metadata": {"chunk_id": "c1"}},
                {"text": "Paragraph 2", "score": 0.3, "metadata": {"chunk_id": "c2"}},
            ]

    class StubReranker:
        async def rerank(self, query, items, top_k=None):
            assert query == "pricing"
            return [items[0]]  # trim to top 1

    service.document_processor = StubDocumentProcessor()
    service.reranker = StubReranker()

    results = await service.retrieve_context("pricing", agent_id=9, top_k=2, return_full=True)
    assert len(results) == 1
    assert results[0]["text"] == "Paragraph 1"
    assert results[0]["chunk_id"] == "c1"

    simple = await service.retrieve_context("pricing", agent_id=9, top_k=2)
    assert simple == ["Paragraph 1"]
