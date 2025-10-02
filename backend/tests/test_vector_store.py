from pathlib import Path
import sys
import types
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.mark.asyncio
async def test_vector_store_add_vectors_without_pinecone(monkeypatch):
    from app.services.vector_store import VectorStoreService

    store = VectorStoreService()

    async def fake_initialize():
        return False

    store._initialized = True
    store.pc = None
    store.index = None

    monkeypatch.setattr(store, "_initialize_pinecone", fake_initialize)

    ids = await store.add_vectors([[0.1, 0.2]], ["text"], [{"meta": 1}])
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_vector_store_search_with_mock_index(monkeypatch):
    from app.services.vector_store import VectorStoreService

    store = VectorStoreService()

    class MockMatch:
        def __init__(self, text, score, metadata):
            self.metadata = {"text": text}
            self.metadata.update(metadata)
            self.score = score

    class MockIndex:
        def query(self, vector, top_k, include_metadata, filter):
            assert filter == {"agent_id": 5}
            return types.SimpleNamespace(matches=[
                MockMatch("chunk", 0.8, {"source": "doc"}),
                MockMatch("chunk2", 0.6, {"source": "doc2"}),
            ])

    async def fake_initialize():
        store.index = MockIndex()
        return True

    monkeypatch.setattr(store, "_initialize_pinecone", fake_initialize)

    results = await store.search_similar([0.1, 0.2], agent_id=5, top_k=2, score_threshold=0.7)
    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "doc"
