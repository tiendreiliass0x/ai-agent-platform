"""
Vector Store Abstraction Layer

This module provides a common interface for different vector database backends,
allowing easy switching between providers like Pinecone, pgvector, Milvus, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from enum import Enum


class VectorStoreType(str, Enum):
    PINECONE = "pinecone"
    PGVECTOR = "pgvector"
    MILVUS = "milvus"
    REDIS = "redis"
    QDRANT = "qdrant"
    CHROMA = "chroma"


@dataclass
class VectorSearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class VectorStoreConfig:
    store_type: VectorStoreType
    connection_params: Dict[str, Any]
    index_name: str
    dimension: int = 3072
    metric: str = "cosine"


class VectorStoreInterface(ABC):
    """Abstract base class for vector store implementations"""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the vector store connection and index"""
        pass

    @abstractmethod
    async def add_vectors(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Add vectors to the store"""
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        agent_id: int,
        top_k: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        pass

    @abstractmethod
    async def delete_by_filter(self, filters: Dict[str, Any]) -> bool:
        """Delete vectors matching filters"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy"""
        pass


class VectorStoreFactory:
    """Factory for creating vector store instances"""

    @staticmethod
    def create_store(config: VectorStoreConfig) -> VectorStoreInterface:
        """Create a vector store instance based on configuration"""

        if config.store_type == VectorStoreType.PINECONE:
            from .vector_stores.pinecone_store import PineconeVectorStore
            return PineconeVectorStore(config)

        elif config.store_type == VectorStoreType.PGVECTOR:
            from .vector_stores.pgvector_store import PgVectorStore
            return PgVectorStore(config)

        elif config.store_type == VectorStoreType.MILVUS:
            from .vector_stores.milvus_store import MilvusVectorStore
            return MilvusVectorStore(config)

        elif config.store_type == VectorStoreType.REDIS:
            from .vector_stores.redis_store import RedisVectorStore
            return RedisVectorStore(config)

        elif config.store_type == VectorStoreType.QDRANT:
            from .vector_stores.qdrant_store import QdrantVectorStore
            return QdrantVectorStore(config)

        elif config.store_type == VectorStoreType.CHROMA:
            from .vector_stores.chroma_store import ChromaVectorStore
            return ChromaVectorStore(config)

        else:
            raise ValueError(f"Unsupported vector store type: {config.store_type}")


class PerformanceMetrics:
    """Performance tracking for vector operations"""

    def __init__(self):
        self.search_times: List[float] = []
        self.insert_times: List[float] = []
        self.delete_times: List[float] = []

    def record_search(self, duration: float):
        self.search_times.append(duration)

    def record_insert(self, duration: float):
        self.insert_times.append(duration)

    def record_delete(self, duration: float):
        self.delete_times.append(duration)

    def get_average_search_time(self) -> float:
        return sum(self.search_times) / len(self.search_times) if self.search_times else 0.0

    def get_average_insert_time(self) -> float:
        return sum(self.insert_times) / len(self.insert_times) if self.insert_times else 0.0

    def get_average_delete_time(self) -> float:
        return sum(self.delete_times) / len(self.delete_times) if self.delete_times else 0.0


class VectorStoreManager:
    """High-level manager for vector store operations with performance tracking"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.store = VectorStoreFactory.create_store(config)
        self.metrics = PerformanceMetrics()
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the vector store"""
        if not self._initialized:
            self._initialized = await self.store.initialize()
        return self._initialized

    async def add_vectors_with_tracking(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Add vectors with performance tracking"""
        import time
        start_time = time.time()

        try:
            result = await self.store.add_vectors(embeddings, texts, metadatas)
            duration = time.time() - start_time
            self.metrics.record_insert(duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_insert(duration)
            raise

    async def search_with_tracking(
        self,
        query_embedding: List[float],
        agent_id: int,
        top_k: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search with performance tracking"""
        import time
        start_time = time.time()

        try:
            result = await self.store.search_similar(
                query_embedding, agent_id, top_k, score_threshold, filters
            )
            duration = time.time() - start_time
            self.metrics.record_search(duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_search(duration)
            raise

    async def delete_with_tracking(self, vector_ids: List[str]) -> bool:
        """Delete vectors with performance tracking"""
        import time
        start_time = time.time()

        try:
            result = await self.store.delete_vectors(vector_ids)
            duration = time.time() - start_time
            self.metrics.record_delete(duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_delete(duration)
            raise

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return {
            "store_type": self.config.store_type.value,
            "average_search_time_ms": self.metrics.get_average_search_time() * 1000,
            "average_insert_time_ms": self.metrics.get_average_insert_time() * 1000,
            "average_delete_time_ms": self.metrics.get_average_delete_time() * 1000,
            "total_searches": len(self.metrics.search_times),
            "total_inserts": len(self.metrics.insert_times),
            "total_deletes": len(self.metrics.delete_times)
        }