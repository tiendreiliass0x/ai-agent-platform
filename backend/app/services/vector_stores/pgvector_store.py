"""
PostgreSQL with pgvector extension implementation for vector storage.

This implementation leverages the existing PostgreSQL infrastructure
to provide vector search capabilities with ACID guarantees and SQL integration.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy import text, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from ..vector_store_base import VectorStoreInterface, VectorSearchResult, VectorStoreConfig
from ...core.database import get_async_session, Base


class VectorDocument(Base):
    """SQLAlchemy model for vector documents"""
    __tablename__ = "vector_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(Integer, nullable=False, index=True)
    text_content = Column(Text, nullable=False)
    embedding = Column("embedding", nullable=False)  # Will be VECTOR type
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        # Add indexes for common query patterns
        {"schema": None}
    )


class PgVectorStore(VectorStoreInterface):
    """PostgreSQL with pgvector extension vector store implementation"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.dimension = config.dimension
        self.session: Optional[AsyncSession] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize pgvector extension and create tables"""
        if self._initialized:
            return True

        try:
            self.session = await get_async_session()

            # Enable pgvector extension
            await self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Create tables with vector column
            await self.session.execute(text(f"""
                CREATE TABLE IF NOT EXISTS vector_documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    agent_id INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    embedding vector({self.dimension}) NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))

            # Create indexes for performance
            await self.session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_vector_documents_agent_id
                ON vector_documents(agent_id)
            """))

            # Create vector similarity index using HNSW
            await self.session.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_vector_documents_embedding_cosine
                ON vector_documents USING hnsw (embedding vector_cosine_ops)
            """))

            # Create GIN index for metadata queries
            await self.session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_vector_documents_metadata
                ON vector_documents USING gin(metadata)
            """))

            await self.session.commit()
            self._initialized = True
            return True

        except Exception as e:
            print(f"Error initializing pgvector store: {e}")
            if self.session:
                await self.session.rollback()
            return False

    async def add_vectors(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Add vectors to PostgreSQL with pgvector"""
        if not self._initialized:
            await self.initialize()

        session = await get_async_session()
        vector_ids = []

        try:
            for embedding, text, metadata in zip(embeddings, texts, metadatas):
                vector_id = str(uuid.uuid4())
                vector_ids.append(vector_id)

                # Convert embedding to pgvector format
                embedding_str = f"[{','.join(map(str, embedding))}]"

                await session.execute(text("""
                    INSERT INTO vector_documents (id, agent_id, text_content, embedding, metadata)
                    VALUES (:id, :agent_id, :text_content, :embedding, :metadata)
                """), {
                    "id": vector_id,
                    "agent_id": metadata.get("agent_id"),
                    "text_content": text,
                    "embedding": embedding_str,
                    "metadata": metadata
                })

            await session.commit()
            return vector_ids

        except Exception as e:
            await session.rollback()
            print(f"Error adding vectors to pgvector: {e}")
            raise
        finally:
            await session.close()

    async def search_similar(
        self,
        query_embedding: List[float],
        agent_id: int,
        top_k: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using cosine similarity"""
        if not self._initialized:
            await self.initialize()

        session = await get_async_session()

        try:
            # Convert query embedding to pgvector format
            query_embedding_str = f"[{','.join(map(str, query_embedding))}]"

            # Build the query with optional metadata filters
            base_query = """
                SELECT
                    id,
                    text_content,
                    metadata,
                    1 - (embedding <=> :query_embedding) as similarity_score
                FROM vector_documents
                WHERE agent_id = :agent_id
            """

            params = {
                "query_embedding": query_embedding_str,
                "agent_id": agent_id
            }

            # Add metadata filters if provided
            if filters:
                for key, value in filters.items():
                    if key != "agent_id":  # agent_id already handled
                        base_query += f" AND metadata->>'{key}' = :{key}"
                        params[key] = str(value)

            # Complete the query
            final_query = f"""
                {base_query}
                AND (1 - (embedding <=> :query_embedding)) >= :score_threshold
                ORDER BY embedding <=> :query_embedding
                LIMIT :top_k
            """

            params.update({
                "score_threshold": score_threshold,
                "top_k": top_k
            })

            result = await session.execute(text(final_query), params)
            rows = result.fetchall()

            search_results = []
            for row in rows:
                search_results.append(VectorSearchResult(
                    id=str(row.id),
                    text=row.text_content,
                    score=float(row.similarity_score),
                    metadata=row.metadata or {}
                ))

            return search_results

        except Exception as e:
            print(f"Error searching vectors in pgvector: {e}")
            return []
        finally:
            await session.close()

    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by their IDs"""
        if not self._initialized:
            await self.initialize()

        session = await get_async_session()

        try:
            # Convert UUIDs to proper format for the query
            id_placeholders = ",".join([f"'{vid}'" for vid in vector_ids])

            await session.execute(text(f"""
                DELETE FROM vector_documents
                WHERE id IN ({id_placeholders})
            """))

            await session.commit()
            return True

        except Exception as e:
            await session.rollback()
            print(f"Error deleting vectors from pgvector: {e}")
            return False
        finally:
            await session.close()

    async def delete_by_filter(self, filters: Dict[str, Any]) -> bool:
        """Delete vectors matching filters"""
        if not self._initialized:
            await self.initialize()

        session = await get_async_session()

        try:
            query = "DELETE FROM vector_documents WHERE 1=1"
            params = {}

            for key, value in filters.items():
                if key == "agent_id":
                    query += " AND agent_id = :agent_id"
                    params["agent_id"] = value
                else:
                    query += f" AND metadata->>'{key}' = :{key}"
                    params[key] = str(value)

            await session.execute(text(query), params)
            await session.commit()
            return True

        except Exception as e:
            await session.rollback()
            print(f"Error deleting vectors by filter from pgvector: {e}")
            return False
        finally:
            await session.close()

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self._initialized:
            await self.initialize()

        session = await get_async_session()

        try:
            # Get total count
            result = await session.execute(text("""
                SELECT COUNT(*) as total_vectors,
                       COUNT(DISTINCT agent_id) as unique_agents,
                       AVG(length(text_content)) as avg_text_length
                FROM vector_documents
            """))
            stats = result.fetchone()

            # Get storage size
            size_result = await session.execute(text("""
                SELECT pg_size_pretty(pg_total_relation_size('vector_documents')) as table_size,
                       pg_size_pretty(pg_relation_size('vector_documents')) as data_size
            """))
            size_stats = size_result.fetchone()

            return {
                "status": "available",
                "total_vectors": stats.total_vectors,
                "unique_agents": stats.unique_agents,
                "avg_text_length": float(stats.avg_text_length) if stats.avg_text_length else 0,
                "table_size": size_stats.table_size,
                "data_size": size_stats.data_size,
                "dimension": self.dimension
            }

        except Exception as e:
            print(f"Error getting pgvector stats: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            await session.close()

    async def health_check(self) -> bool:
        """Check if pgvector is healthy"""
        try:
            session = await get_async_session()

            # Test pgvector extension
            await session.execute(text("SELECT 1"))

            # Test vector operations
            await session.execute(text("""
                SELECT '[1,2,3]'::vector <=> '[3,2,1]'::vector as test_similarity
            """))

            await session.close()
            return True

        except Exception as e:
            print(f"pgvector health check failed: {e}")
            return False

    async def optimize_index(self) -> bool:
        """Optimize vector indexes for better performance"""
        if not self._initialized:
            await self.initialize()

        session = await get_async_session()

        try:
            # Update table statistics
            await session.execute(text("ANALYZE vector_documents"))

            # Rebuild indexes if needed (optional)
            await session.execute(text("REINDEX INDEX idx_vector_documents_embedding_cosine"))

            await session.commit()
            return True

        except Exception as e:
            await session.rollback()
            print(f"Error optimizing pgvector indexes: {e}")
            return False
        finally:
            await session.close()

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        agent_id: int,
        top_k: int = 5,
        text_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining full-text search and vector similarity.
        This is a key advantage of pgvector over other solutions.
        """
        if not self._initialized:
            await self.initialize()

        session = await get_async_session()

        try:
            query_embedding_str = f"[{','.join(map(str, query_embedding))}]"

            # Hybrid search using both text similarity and vector similarity
            hybrid_query = """
                WITH text_search AS (
                    SELECT
                        id,
                        text_content,
                        metadata,
                        ts_rank(to_tsvector('english', text_content), plainto_tsquery('english', :query_text)) as text_score
                    FROM vector_documents
                    WHERE agent_id = :agent_id
                    AND to_tsvector('english', text_content) @@ plainto_tsquery('english', :query_text)
                ),
                vector_search AS (
                    SELECT
                        id,
                        text_content,
                        metadata,
                        1 - (embedding <=> :query_embedding) as vector_score
                    FROM vector_documents
                    WHERE agent_id = :agent_id
                ),
                combined AS (
                    SELECT
                        COALESCE(t.id, v.id) as id,
                        COALESCE(t.text_content, v.text_content) as text_content,
                        COALESCE(t.metadata, v.metadata) as metadata,
                        COALESCE(t.text_score, 0) * :text_weight + COALESCE(v.vector_score, 0) * :vector_weight as combined_score
                    FROM text_search t
                    FULL OUTER JOIN vector_search v ON t.id = v.id
                )
                SELECT * FROM combined
                WHERE combined_score > 0
                ORDER BY combined_score DESC
                LIMIT :top_k
            """

            result = await session.execute(text(hybrid_query), {
                "query_text": query_text,
                "query_embedding": query_embedding_str,
                "agent_id": agent_id,
                "text_weight": text_weight,
                "vector_weight": vector_weight,
                "top_k": top_k
            })

            rows = result.fetchall()

            search_results = []
            for row in rows:
                search_results.append(VectorSearchResult(
                    id=str(row.id),
                    text=row.text_content,
                    score=float(row.combined_score),
                    metadata=row.metadata or {}
                ))

            return search_results

        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []
        finally:
            await session.close()