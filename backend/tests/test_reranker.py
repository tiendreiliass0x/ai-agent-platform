from pathlib import Path
import sys
import types

import pytest
from unittest.mock import AsyncMock

if 'huggingface_hub' not in sys.modules:
    hub_stub = types.ModuleType('huggingface_hub')
    hub_stub.HfApi = object
    hub_stub.HfFolder = object
    hub_stub.Repository = object
    hub_stub.hf_hub_url = lambda *args, **kwargs: ""
    hub_stub.cached_download = lambda *args, **kwargs: ""
    sys.modules['huggingface_hub'] = hub_stub

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
            return [0.0 for _ in pairs]

    st_stub.SentenceTransformer = _DummySentenceTransformer
    st_stub.CrossEncoder = _DummyCrossEncoder
    sys.modules['sentence_transformers'] = st_stub

if 'app.services.embedding_service' not in sys.modules:
    emb_stub = types.ModuleType('app.services.embedding_service')

    class _DummyEmbeddingService:
        def __init__(self, *args, **kwargs):
            pass

        async def generate_embeddings(self, texts):
            return [[0.0] * 4 for _ in texts]

    class _DummyEmbeddingCache:
        def __init__(self, *args, **kwargs):
            self._store = {}

        def get(self, text):
            return None

        def set(self, text, embedding):
            self._store[text] = embedding

        def clear(self):
            self._store.clear()

    emb_stub.EmbeddingService = _DummyEmbeddingService
    emb_stub.EmbeddingCache = _DummyEmbeddingCache
    sys.modules['app.services.embedding_service'] = emb_stub

if 'app.services.document_processor' not in sys.modules:
    doc_stub = types.ModuleType('app.services.document_processor')

    class _DummyDocumentProcessor:
        def __init__(self, *args, **kwargs):
            pass

        async def search_similar_content(self, query, agent_id, top_k):
            return []

    doc_stub.DocumentProcessor = _DummyDocumentProcessor
    sys.modules['app.services.document_processor'] = doc_stub

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.reranker_service import RerankerService
from app.services import rag_service
from app.services import domain_expertise_service
from app.services.domain_expertise_service import DomainExpertiseService, RetrievalCandidate


class DummyModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return list(self._scores)


@pytest.mark.asyncio
async def test_reranker_service_sorts_by_model_scores():
    reranker = RerankerService()
    reranker._model = DummyModel(scores=[0.2, 0.85])

    items = [
        {"text": "First passage", "score": 0.4, "metadata": {"id": "a"}},
        {"text": "Second passage", "score": 0.1, "metadata": {"id": "b"}},
    ]

    ranked = await reranker.rerank("example question", items)

    assert ranked[0]["text"] == "Second passage"
    assert ranked[0]["rerank_score"] == pytest.approx(0.85)
    assert ranked[0]["combined_score"] > ranked[1]["combined_score"]


@pytest.mark.asyncio
async def test_reranker_service_returns_original_when_model_missing(monkeypatch):
    reranker = RerankerService()
    reranker._model = None
    monkeypatch.setattr(reranker, "_get_model", AsyncMock(return_value=None))

    items = [
        {"text": "Alpha", "score": 0.3},
        {"text": "Beta", "score": 0.2},
    ]

    ranked = await reranker.rerank("query", items)
    assert ranked == items


@pytest.mark.asyncio
async def test_rag_service_uses_reranker(monkeypatch):
    service = rag_service.RAGService()

    class StubDocumentProcessor:
        async def search_similar_content(self, query, agent_id, top_k):
            return [
                {"text": "Doc A", "score": 0.1, "metadata": {"source": "a"}},
                {"text": "Doc B", "score": 0.4, "metadata": {"source": "b"}},
            ]

    class StubReranker:
        async def rerank(self, query, items, top_k=None):
            assert query == "sample"
            return list(reversed(items))

    service.document_processor = StubDocumentProcessor()
    service.reranker = StubReranker()

    results = await service.retrieve_context("sample", agent_id=99, top_k=2)
    assert results == ["Doc B", "Doc A"]

    full_results = await service.retrieve_context("sample", agent_id=99, top_k=2, return_full=True)
    assert full_results[0]["text"] == "Doc B"


@pytest.mark.asyncio
async def test_domain_expertise_rerank_uses_shared_service(monkeypatch):
    service = DomainExpertiseService()

    candidates = [
        RetrievalCandidate(doc_id="a", content="Result A", score=0.2),
        RetrievalCandidate(doc_id="b", content="Result B", score=0.3),
    ]

    class DummyReranker:
        async def rerank(self, query, items, top_k=None):
            assert query == "customer question"
            return [
                {**items[1], "combined_score": 0.9},
                {**items[0], "combined_score": 0.2},
            ]

    monkeypatch.setattr(domain_expertise_service, "reranker_service", DummyReranker())

    ranked = await service._rerank_candidates(candidates, "customer question", knowledge_pack=None)

    assert [c.doc_id for c in ranked] == ["b", "a"]
    assert ranked[0].score > ranked[1].score
