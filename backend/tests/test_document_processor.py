from pathlib import Path
import sys
import types
import pytest
import asyncio

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    if 'semantic_text_splitter' not in sys.modules:
        sts = types.ModuleType('semantic_text_splitter')

        class TextSplitter:
            @classmethod
            def from_language(cls, language, chunk_size, chunk_overlap):
                return cls()

            def split_text(self, text):
                return [text]

        sts.TextSplitter = TextSplitter
        sys.modules['semantic_text_splitter'] = sts

    if 'yake' not in sys.modules:
        yake_stub = types.ModuleType('yake')

        class KeywordExtractor:
            def __init__(self, *args, **kwargs):
                pass

            def extract_keywords(self, text):
                return [("keyword", 0.1)]

        yake_stub.KeywordExtractor = KeywordExtractor
        sys.modules['yake'] = yake_stub

    if 'firecrawl' not in sys.modules:
        firecrawl_stub = types.ModuleType('firecrawl')

        class FirecrawlApp:
            def __init__(self, api_key):
                pass

        firecrawl_stub.FirecrawlApp = FirecrawlApp
        sys.modules['firecrawl'] = firecrawl_stub


@pytest.mark.asyncio
async def test_document_processor_search_similar_content(monkeypatch):
    from app.services.document_processor import DocumentProcessor

    processor = DocumentProcessor()

    class VectorStoreStub:
        async def search_similar(self, query_embedding, agent_id, top_k=5, score_threshold=0.7):
            return [
                {"text": "chunk one", "score": 0.9, "metadata": {"chunk_id": "1"}},
                {"text": "chunk two", "score": 0.8, "metadata": {"chunk_id": "2"}},
            ]

    async def fake_generate_embeddings(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    processor.vector_store = VectorStoreStub()
    processor.embedding_service.generate_embeddings = fake_generate_embeddings

    results = await processor.search_similar_content("query", agent_id=7, top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "chunk one"
    assert results[1]["metadata"]["chunk_id"] == "2"


@pytest.mark.asyncio
async def test_document_processor_process_text(monkeypatch, tmp_path):
    from app.services.document_processor import DocumentProcessor

    processor = DocumentProcessor()

    class VectorStoreStub:
        async def add_vectors(self, embeddings, texts, metadatas):
            assert len(embeddings) == len(texts) == len(metadatas)
            return [f"vec_{i}" for i in range(len(embeddings))]

    async def fake_generate_embeddings(texts):
        return [[0.1, 0.1, 0.1] for _ in texts]

    processor.vector_store = VectorStoreStub()
    processor.embedding_service.generate_embeddings = fake_generate_embeddings

    result = await processor._process_text(
        text_content="Example content for splitting",
        agent_id=10,
        source="source.txt",
        document_id=99,
        extra_metadata=None,
        base_metadata={"content_type": "text/plain"}
    )

    assert result["status"] == "completed"
    assert "chunk_count" in result
    assert result["vector_ids"] == ["vec_0"]
