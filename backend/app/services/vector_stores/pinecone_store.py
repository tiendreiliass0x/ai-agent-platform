"""
Pinecone vector store implementation using the abstraction layer.
"""

import uuid
import asyncio
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

from ..vector_store_base import VectorStoreInterface, VectorSearchResult, VectorStoreConfig


class PineconeVectorStore(VectorStoreInterface):
    """Pinecone implementation of the vector store interface"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.pc = None
        self.index = None
        self._initialized = False

        # Extract Pinecone-specific configuration
        self.api_key = config.connection_params.get("api_key")
        self.environment = config.connection_params.get("environment", "us-east-1")
        self.cloud = config.connection_params.get("cloud", "aws")

    async def initialize(self) -> bool:
        """Initialize Pinecone client and index"""
        if self._initialized:
            return True

        if not self.api_key:
            print("Warning: Pinecone API key not provided")
            self._initialized = True
            return False

        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not (run in executor to avoid blocking)
            existing_indexes = await asyncio.get_event_loop().run_in_executor(
                None, lambda: [index.name for index in self.pc.list_indexes()]
            )

            if self.config.index_name not in existing_indexes:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.pc.create_index(
                        name=self.config.index_name,
                        dimension=self.config.dimension,
                        metric=self.config.metric,
                        spec=ServerlessSpec(
                            cloud=self.cloud,
                            region=self.environment
                        )
                    )
                )

            self.index = self.pc.Index(self.config.index_name)
            self._initialized = True
            return True

        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            self._initialized = True
            return False

    async def add_vectors(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Add vectors to Pinecone"""
        if not self._initialized:
            if not await self.initialize():
                return []

        try:
            vectors = []
            vector_ids = []

            for embedding, text, metadata in zip(embeddings, texts, metadatas):
                vector_id = str(uuid.uuid4())
                vector_ids.append(vector_id)

                # Add text to metadata for retrieval
                metadata_with_text = metadata.copy()
                metadata_with_text["text"] = text

                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata_with_text
                })

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda b=batch: self.index.upsert(vectors=b)
                )

            return vector_ids

        except Exception as e:
            print(f"Error adding vectors to Pinecone: {e}")
            raise

    async def search_similar(
        self,
        query_embedding: List[float],
        agent_id: int,
        top_k: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in Pinecone"""
        if not self._initialized:
            if not await self.initialize():
                return []

        try:
            # Build filter dictionary
            filter_dict = {"agent_id": agent_id}
            if filters:
                filter_dict.update(filters)

            # Query Pinecone
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
            )

            results = []
            for match in response.matches:
                if match.score >= score_threshold:
                    metadata = match.metadata.copy()
                    text = metadata.pop("text", "")

                    results.append(VectorSearchResult(
                        id=match.id,
                        text=text,
                        score=match.score,
                        metadata=metadata
                    ))

            return results

        except Exception as e:
            print(f"Error searching vectors in Pinecone: {e}")
            return []

    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors from Pinecone"""
        if not self._initialized:
            if not await self.initialize():
                return False

        try:
            # Delete in batches
            batch_size = 1000
            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i:i + batch_size]
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda b=batch: self.index.delete(ids=b)
                )
            return True

        except Exception as e:
            print(f"Error deleting vectors from Pinecone: {e}")
            return False

    async def delete_by_filter(self, filters: Dict[str, Any]) -> bool:
        """Delete vectors matching filters"""
        if not self._initialized:
            if not await self.initialize():
                return False

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.delete(filter=filters)
            )
            return True

        except Exception as e:
            print(f"Error deleting vectors by filter from Pinecone: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self._initialized:
            if not await self.initialize():
                return {"status": "unavailable"}

        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.describe_index_stats
            )

            return {
                "status": "available",
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }

        except Exception as e:
            print(f"Error getting Pinecone stats: {e}")
            return {"status": "error", "error": str(e)}

    async def health_check(self) -> bool:
        """Check if Pinecone is healthy"""
        try:
            if not self._initialized:
                return await self.initialize()

            # Try to get index stats as a health check
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.describe_index_stats
            )
            return True

        except Exception as e:
            print(f"Pinecone health check failed: {e}")
            return False