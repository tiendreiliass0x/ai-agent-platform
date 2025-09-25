import asyncio
import aiofiles
from typing import List, Dict, Any, Optional
import hashlib
import uuid

from app.utils.text_processing import SimpleTextSplitter, SimpleDocument
from bs4 import BeautifulSoup
import requests
import pypdf

from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.embedding_service import EmbeddingService

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = SimpleTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = VectorStoreService()
        self.embedding_service = EmbeddingService()

    async def process_file(
        self,
        file_path: str,
        agent_id: int,
        filename: str,
        file_type: str,
        *,
        document_id: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process uploaded file and create embeddings"""
        try:
            # Extract text based on file type
            if file_type == "application/pdf":
                text_content = await self._extract_pdf_text(file_path)
            elif file_type == "text/html":
                text_content = await self._extract_html_text(file_path)
            elif file_type == "text/plain":
                text_content = await self._extract_text_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Create chunks
            chunks = await self._create_chunks(text_content, filename)

            # Generate embeddings and store in vector database
            return await self._process_text(
                text_content=text_content,
                agent_id=agent_id,
                source=filename,
                document_id=document_id,
                extra_metadata=extra_metadata
            )

        except Exception as e:
            return self._build_failure_response(str(e))

    async def process_url(
        self,
        url: str,
        agent_id: int,
        *,
        document_id: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process webpage URL and create embeddings"""
        try:
            # Fetch webpage content
            text_content = await self._fetch_webpage_content(url)

            return await self._process_text(
                text_content=text_content,
                agent_id=agent_id,
                source=url,
                document_id=document_id,
                extra_metadata=extra_metadata
            )

        except Exception as e:
            return self._build_failure_response(str(e))

    async def process_text_content(
        self,
        text: str,
        agent_id: int,
        *,
        source: str,
        document_id: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process raw text content and create embeddings."""
        if not text:
            return self._build_failure_response("No content provided for processing")

        return await self._process_text(
            text_content=text,
            agent_id=agent_id,
            source=source,
            document_id=document_id,
            extra_metadata=extra_metadata
        )

    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        async with aiofiles.open(file_path, 'rb') as file:
            pdf_content = await file.read()

        # Use pypdf for better async support
        import io
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_content))

        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        return text.strip()

    async def _extract_html_text(self, file_path: str) -> str:
        """Extract text from HTML file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            html_content = await file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    async def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()

    async def _fetch_webpage_content(self, url: str) -> str:
        """Fetch and extract text from webpage"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    async def _create_chunks(self, text: str, source: str) -> List[SimpleDocument]:
        """Split text into chunks for embedding"""
        chunks = self.text_splitter.split_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = SimpleDocument(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "chunk_id": str(uuid.uuid4())
                }
            )
            documents.append(doc)

        return documents

    async def _store_chunks(
        self,
        chunks: List[SimpleDocument],
        agent_id: int,
        *,
        document_id: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate embeddings and store in vector database"""
        # Generate embeddings for all chunks
        texts = [doc.page_content for doc in chunks]
        embeddings = await self.embedding_service.generate_embeddings(texts)

        # Prepare metadata for each chunk
        metadatas = []
        for doc in chunks:
            metadata = doc.metadata.copy()
            metadata["agent_id"] = agent_id
            if document_id is not None:
                metadata["document_id"] = document_id
            if extra_metadata:
                metadata.update(extra_metadata)
            metadatas.append(metadata)

        # Store in vector database
        vector_ids = await self.vector_store.add_vectors(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas
        )

        return vector_ids

    async def _process_text(
        self,
        text_content: str,
        agent_id: int,
        *,
        source: str,
        document_id: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            chunks = await self._create_chunks(text_content, source)
            vector_ids = await self._store_chunks(
                chunks,
                agent_id,
                document_id=document_id,
                extra_metadata=extra_metadata
            )

            return {
                "status": "completed",
                "chunk_count": len(chunks),
                "vector_ids": vector_ids,
                "preview": text_content[:500],
                "extracted_text": text_content
            }
        except Exception as exc:
            return self._build_failure_response(str(exc), text_content=text_content)

    def _build_failure_response(
        self,
        error_message: str,
        *,
        text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "status": "failed",
            "error_message": error_message,
            "chunk_count": 0,
            "vector_ids": [],
            "preview": (text_content[:500] if text_content else ""),
            "extracted_text": text_content or ""
        }

    async def search_similar_content(
        self,
        query: str,
        agent_id: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity"""
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embeddings([query])

        # Search in vector store
        results = await self.vector_store.search_similar(
            query_embedding=query_embedding[0],
            agent_id=agent_id,
            top_k=top_k
        )

        return results

    async def delete_document_vectors(self, vector_ids: List[str]):
        """Delete vectors for a document"""
        await self.vector_store.delete_vectors(vector_ids)
