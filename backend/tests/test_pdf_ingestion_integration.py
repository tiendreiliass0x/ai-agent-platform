import pytest
from pathlib import Path


@pytest.mark.asyncio
async def test_pdf_ingestion_process_file(monkeypatch, tmp_path):
    from app.services.document_processor import DocumentProcessor

    processor = DocumentProcessor()

    # Stub embeddings and vector store to avoid external deps
    async def fake_generate_embeddings(texts):
        return [[0.1, 0.1, 0.1] for _ in texts]

    async def fake_add_vectors(embeddings, texts, metadatas):
        return [f"vec_{i}" for i in range(len(embeddings))]

    processor.embedding_service.generate_embeddings = fake_generate_embeddings
    processor.vector_store.add_vectors = fake_add_vectors  # type: ignore

    # Patch PDF extraction to avoid heavy libs
    async def fake_extract_pdf_text(file_path: str):
        return "Sample PDF text content for testing.", {"content_type": "application/pdf"}

    monkeypatch.setattr(processor, "_extract_pdf_text", fake_extract_pdf_text)

    # Create a tiny fake PDF file on disk
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% Fake PDF content for testing\n")

    result = await processor.process_file(
        file_path=str(pdf_path),
        agent_id=1,
        filename="sample.pdf",
        file_type="application/pdf",
        document_id=42,
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] > 0
    assert len(result["vector_ids"]) == result["chunk_count"]
    assert result["metadata"].get("content_type") == "application/pdf"

