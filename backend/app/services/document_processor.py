import asyncio
import aiofiles
import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

try:  # Optional imports
    from firecrawl import FirecrawlApp  # type: ignore
except ImportError:  # pragma: no cover
    FirecrawlApp = None  # type: ignore

try:  # pragma: no cover
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

try:  # pragma: no cover
    import htmldate
except ImportError:  # pragma: no cover
    htmldate = None

try:  # pragma: no cover
    import extruct
except ImportError:  # pragma: no cover
    extruct = None

try:  # pragma: no cover
    from pymupdf4llm import to_markdown
except ImportError:  # pragma: no cover
    to_markdown = None

try:  # pragma: no cover
    from simhash import Simhash
except ImportError:  # pragma: no cover
    Simhash = None

try:  # pragma: no cover
    from semantic_text_splitter import TextSplitter as SemanticTextSplitter
except ImportError:  # pragma: no cover
    SemanticTextSplitter = None

try:  # pragma: no cover
    import yake
except ImportError:  # pragma: no cover
    yake = None

from app.utils.text_processing import SimpleTextSplitter, SimpleDocument
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.embedding_service import EmbeddingService

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.semantic_splitter = None
        if SemanticTextSplitter is not None:
            try:
                self.semantic_splitter = SemanticTextSplitter.from_language(
                    language="en",
                    chunk_size=600,
                    chunk_overlap=100
                )
            except Exception:
                self.semantic_splitter = None

        self.vector_store = VectorStoreService()
        self.embedding_service = EmbeddingService()

        self.firecrawl_app = None
        if FirecrawlApp and settings.FIRECRAWL_API_KEY:
            try:
                self.firecrawl_app = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)
            except Exception:
                self.firecrawl_app = None

        self.keyword_extractor = None
        if yake is not None:
            try:
                self.keyword_extractor = yake.KeywordExtractor(lan="en", top=5)
            except Exception:
                self.keyword_extractor = None

        self.simhash_threshold = 3

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
            base_metadata: Dict[str, Any] = {}
            if file_type == "application/pdf":
                text_content, pdf_meta = await self._extract_pdf_text(file_path)
                base_metadata.update(pdf_meta)
            elif file_type == "text/html":
                text_content, html_meta = await self._extract_html_text(file_path)
                base_metadata.update(html_meta)
            elif file_type == "text/plain":
                text_content = await self._extract_text_file(file_path)
                base_metadata.update({"content_type": "text/plain"})
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            base_metadata.setdefault("source_url", filename)
            base_metadata.setdefault("title", Path(filename).stem)

            return await self._process_text(
                text_content=text_content,
                agent_id=agent_id,
                source=filename,
                document_id=document_id,
                extra_metadata=extra_metadata,
                base_metadata=base_metadata
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
            text_content, page_metadata = await self._fetch_webpage_content(url)

            return await self._process_text(
                text_content=text_content,
                agent_id=agent_id,
                source=url,
                document_id=document_id,
                extra_metadata=extra_metadata,
                base_metadata=page_metadata
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
            extra_metadata=extra_metadata,
            base_metadata={"source_url": source, "content_type": "text/plain"}
        )

    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        async with aiofiles.open(file_path, 'rb') as file:
            pdf_bytes = await file.read()

        def _extract_text_sync() -> Dict[str, Any]:
            if to_markdown is not None:
                try:
                    markdown = to_markdown(file_path)
                    return {"text": markdown, "metadata": {"content_type": "application/pdf"}}
                except Exception:
                    pass

            import io
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text() or ""
                    text += extracted + "\n"
                return {"text": text.strip(), "metadata": {"content_type": "application/pdf"}}
            except Exception:
                return {"text": "", "metadata": {"content_type": "application/pdf"}}

        result = await asyncio.to_thread(_extract_text_sync)
        return result.get("text", ""), result.get("metadata", {})

    async def _extract_html_text(self, file_path: str) -> str:
        """Extract text from HTML file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            html_content = await file.read()

        def _extract_html_sync():
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            cleaned = '\n'.join(chunk for chunk in lines if chunk)
            title = soup.title.string.strip() if soup.title and soup.title.string else None
            return cleaned, {"title": title, "content_type": "text/html"}

        text, metadata = await asyncio.to_thread(_extract_html_sync)
        return text, metadata

    async def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()

    async def _fetch_webpage_content(self, url: str) -> (str, Dict[str, Any]):
        """Fetch and extract text + metadata from webpage"""
        if self.firecrawl_app is not None:
            try:
                result = await asyncio.to_thread(
                    lambda: self.firecrawl_app.scrape_url(
                        url,
                        params={"formats": ["markdown", "links", "metadata"]}
                    )
                )
                markdown = result.get("markdown") or ""
                metadata = result.get("metadata") or {}
                if metadata:
                    metadata = {
                        "title": metadata.get("title"),
                        "description": metadata.get("description"),
                        "published_at": metadata.get("publishedDate") or metadata.get("published_at"),
                        "source_url": url,
                        "content_type": "text/html"
                    }
                else:
                    metadata = {"source_url": url, "content_type": "text/html"}
                return markdown, metadata
            except Exception:
                pass

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        html = response.text
        text = None
        title = None
        published = None
        schema_entities: List[str] = []

        if trafilatura is not None:
            try:
                downloaded = trafilatura.fetch_url(url) or html
                text = trafilatura.extract(downloaded, include_links=True, include_tables=True)
                metadata = trafilatura.extract_metadata(downloaded)
                if metadata:
                    title = metadata.get("title") or title
                    if metadata.get("date"):
                        published = metadata.get("date")
            except Exception:
                text = None

        if not text:
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            raw_text = soup.get_text(separator="\n")
            lines = (line.strip() for line in raw_text.splitlines())
            text = "\n".join(chunk for chunk in lines if chunk)
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

        if not published and htmldate is not None:
            try:
                published = htmldate.find_date(url)
            except Exception:
                published = None

        if extruct is not None:
            try:
                data = extruct.extract(html, uniform=True)
                for block in (data.get("json-ld") or []):
                    if isinstance(block, dict):
                        schema_type = block.get("@type")
                        if schema_type:
                            schema_entities.append(schema_type)
            except Exception:
                pass

        metadata = {
            "title": title,
            "published_at": published,
            "schema_org_types": schema_entities,
            "source_url": url,
            "content_type": "text/html"
        }

        return text or html, metadata

    async def _create_chunks(self, text: str, base_metadata: Dict[str, Any]) -> List[SimpleDocument]:
        """Split text into enriched chunks for embedding"""
        if not text.strip():
            return []

        if self.semantic_splitter is not None:
            try:
                raw_chunks = self.semantic_splitter.split_text(text)
            except Exception:
                raw_chunks = self.text_splitter.split_text(text)
        else:
            raw_chunks = self.text_splitter.split_text(text)

        documents: List[SimpleDocument] = []
        seen_fingerprints: List[Any] = []

        for index, chunk in enumerate(raw_chunks):
            normalized_chunk = chunk.strip()
            if not normalized_chunk:
                continue

            if Simhash is not None:
                fingerprint = Simhash(normalized_chunk)
                duplicate = False
                for existing in seen_fingerprints:
                    if fingerprint.distance(existing) <= self.simhash_threshold:
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen_fingerprints.append(fingerprint)

            chunk_metadata = self._build_chunk_metadata(
                normalized_chunk,
                index,
                base_metadata
            )

            documents.append(
                SimpleDocument(page_content=normalized_chunk, metadata=chunk_metadata)
            )

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
        try:
            vector_ids = await self.vector_store.add_vectors(
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas
            )
        except Exception as exc:
            print(f"Vector store unavailable, using mock IDs: {exc}")
            vector_ids = [str(uuid.uuid4()) for _ in texts]

        return vector_ids

    async def _process_text(
        self,
        text_content: str,
        agent_id: int,
        *,
        source: str,
        document_id: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            combined_metadata = base_metadata.copy() if base_metadata else {}
            combined_metadata.setdefault("source", source)
            combined_metadata.setdefault("source_url", source)
            combined_metadata.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())

            chunks = await self._create_chunks(text_content, combined_metadata)
            vector_ids = await self._store_chunks(
                chunks,
                agent_id,
                document_id=document_id,
                extra_metadata=combined_metadata if extra_metadata is None else {**combined_metadata, **extra_metadata}
            )

            aggregated_keywords = self._aggregate_keywords(chunks)

            return {
                "status": "completed",
                "chunk_count": len(chunks),
                "vector_ids": vector_ids,
                "preview": text_content[:500],
                "extracted_text": text_content,
                "metadata": combined_metadata,
                "keywords": aggregated_keywords
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
        top_k: int = 5,
        score_threshold: float = 0.7,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity

        Args:
            query: Search query text
            agent_id: Agent ID to filter results
            top_k: Number of top results to return
            score_threshold: Minimum similarity score (0.0-1.0)
            metadata_filters: Optional filters like {"document_type": "manual", "version": "latest"}

        Returns:
            List of search results with text, score, and metadata
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embeddings([query])

        # Search in vector store with filters
        search_kwargs = dict(
            query_embedding=query_embedding[0],
            agent_id=agent_id,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        if metadata_filters:
            search_kwargs["filters"] = metadata_filters

        results = await self.vector_store.search_similar(**search_kwargs)

        return results

    async def delete_document_vectors(self, vector_ids: List[str]):
        """Delete vectors for a document"""
        await self.vector_store.delete_vectors(vector_ids)

    def _build_chunk_metadata(
        self,
        chunk: str,
        index: int,
        base_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        metadata = base_metadata.copy()
        metadata.update({
            "chunk_index": index,
            "chunk_id": str(uuid.uuid4()),
            "hash": hashlib.md5(chunk.encode("utf-8")).hexdigest(),
            "token_count": len(chunk.split()),
            "section_path": self._derive_section_path(chunk),
            "focus_summary": self._summarize_chunk(chunk),
        })

        keywords = self._extract_keywords(chunk)
        if keywords:
            metadata["keywords"] = keywords

        return metadata

    def _derive_section_path(self, chunk: str) -> str:
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped
            if stripped and stripped == stripped.upper() and len(stripped.split()) <= 10:
                return stripped.title()
        return ""

    def _summarize_chunk(self, chunk: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
        summary = " ".join(sentences[:2])
        return summary[:400]

    def _extract_keywords(self, chunk: str) -> Optional[List[str]]:
        if self.keyword_extractor is None:
            return None
        try:
            keywords = self.keyword_extractor.extract_keywords(chunk)
            return [kw for kw, _ in keywords]
        except Exception:
            return None

    def _aggregate_keywords(self, chunks: List[SimpleDocument]) -> List[str]:
        keyword_counts: Dict[str, int] = {}
        for doc in chunks:
            for kw in doc.metadata.get("keywords", []) or []:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_keywords[:10]]
