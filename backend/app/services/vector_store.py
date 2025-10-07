import os
import asyncio
from typing import List, Dict, Any, Optional, Union
import uuid
from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings

class VectorStoreService:
    def __init__(self):
        self.pc = None
        self.index = None
        self._initialized = False

    async def _initialize_pinecone(self):
        """Lazy initialize Pinecone client and index"""
        if self._initialized:
            return True

        if os.environ.get("TESTING") == "1" or not settings.PINECONE_API_KEY:
            print("Warning: Pinecone API key not set. Vector operations will use mock data.")
            self._initialized = True
            return False

        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            # Check if index exists, create if not (this might take time)
            existing_indexes = await asyncio.get_event_loop().run_in_executor(
                None, lambda: [index.name for index in self.pc.list_indexes()]
            )

            if settings.PINECONE_INDEX_NAME not in existing_indexes:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.pc.create_index(
                        name=settings.PINECONE_INDEX_NAME,
                        dimension=3072,  # OpenAI text-embedding-3-large dimension
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                )

            self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
            self._initialized = True
            return True

        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            self.pc = None
            self.index = None
            self._initialized = True
            return False

    async def add_vectors(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Add vectors to the vector store"""
        # Lazy initialization
        if not await self._initialize_pinecone():
            # Return mock IDs if Pinecone is not available
            return [str(uuid.uuid4()) for _ in embeddings]

        try:
            # Prepare vectors for upsert
            vectors = []
            vector_ids = []

            for i, (embedding, text, metadata) in enumerate(zip(embeddings, texts, metadatas)):
                vector_id = str(uuid.uuid4())
                vector_ids.append(vector_id)

                # Add text content to metadata for retrieval
                metadata_with_text = metadata.copy()
                metadata_with_text["text"] = text

                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata_with_text
                })

            # Upsert vectors in batches
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
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        # Lazy initialization
        if not await self._initialize_pinecone():
            # Return mock results if Pinecone is not available
            return [
                {
                    "text": f"Mock result {i+1} for agent {agent_id}",
                    "score": 0.8 - (i * 0.1),
                    "metadata": {"source": "mock", "chunk_index": i}
                }
                for i in range(min(top_k, 3))
            ]

        try:
            # Search with agent_id filter
            vector_filters = {"agent_id": agent_id}
            if filters:
                vector_filters.update(filters)

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=vector_filters
                )
            )

            results = []
            for match in response.matches:
                if match.score >= score_threshold:
                    result = {
                        "text": match.metadata.get("text", ""),
                        "score": match.score,
                        "metadata": {
                            k: v for k, v in match.metadata.items()
                            if k != "text"  # Exclude text from metadata to avoid duplication
                        }
                    }
                    results.append(result)

            return results

        except Exception as e:
            print(f"Error searching vectors in Pinecone: {e}")
            return []

    async def delete_vectors(self, vector_ids: List[str]):
        """Delete vectors by their IDs"""
        if not self.index:
            print("Pinecone not available, skipping vector deletion")
            return

        try:
            # Delete in batches
            batch_size = 1000
            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i:i + batch_size]
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda b=batch: self.index.delete(ids=b)
                )

        except Exception as e:
            print(f"Error deleting vectors from Pinecone: {e}")
            raise

    async def delete_agent_vectors(self, agent_id: int):
        """Delete all vectors for a specific agent"""
        if not self.index:
            print("Pinecone not available, skipping agent vector deletion")
            return

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.delete(filter={"agent_id": agent_id})
            )

        except Exception as e:
            print(f"Error deleting agent vectors from Pinecone: {e}")
            raise

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if not self.index:
            return {"status": "unavailable", "total_vectors": 0}

        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.describe_index_stats
            )

            return {
                "status": "available",
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }

        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {"status": "error", "error": str(e)}
