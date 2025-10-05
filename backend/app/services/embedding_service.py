import asyncio
import hashlib
import json
from typing import List, Any, Optional
import openai
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import settings

# Optional Redis import
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class EmbeddingService:
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = "text-embedding-3-large"
        self.embedding_dimension = 3072

        # Fallback to sentence-transformers if OpenAI is not available
        self.fallback_model = None

        # Initialize Redis cache if available
        self.redis_client = None
        if REDIS_AVAILABLE and settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception as e:
                print(f"Failed to connect to Redis: {e}, falling back to in-memory cache")
                self.redis_client = None

        # In-memory fallback cache
        self.memory_cache = EmbeddingCache()

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with caching"""
        # Check cache first for single queries (optimization for RAG queries)
        if len(texts) == 1:
            cached = await self._get_cached_embedding(texts[0])
            if cached is not None:
                return [cached]

        # Generate embeddings
        try:
            if settings.OPENAI_API_KEY:
                embeddings = await self._generate_openai_embeddings(texts)
            else:
                embeddings = await self._generate_local_embeddings(texts)

            # Cache single query results
            if len(texts) == 1 and len(embeddings) == 1:
                await self._cache_embedding(texts[0], embeddings[0])

            return embeddings

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Fallback to local model
            return await self._generate_local_embeddings(texts)

    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        cache_key = self._get_cache_key(text)

        # Try Redis first
        if self.redis_client:
            try:
                cached_str = await self.redis_client.get(cache_key)
                if cached_str:
                    return json.loads(cached_str)
            except Exception as e:
                print(f"Redis cache get failed: {e}")

        # Fallback to memory cache
        return self.memory_cache.get(text)

    async def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding for text"""
        cache_key = self._get_cache_key(text)

        # Try Redis first (TTL: 1 hour)
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(embedding)
                )
            except Exception as e:
                print(f"Redis cache set failed: {e}")

        # Always cache in memory as fallback
        self.memory_cache.set(text, embedding)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{self.model_name}:{text_hash}"

    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        # Process in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = await self.openai_client.embeddings.create(
                model=self.model_name,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local sentence-transformers model"""
        if self.fallback_model is None:
            self.fallback_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Run in thread pool since sentence-transformers is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.fallback_model.encode,
            texts
        )

        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if settings.OPENAI_API_KEY:
            return self.embedding_dimension
        else:
            # all-MiniLM-L6-v2 has 384 dimensions
            return 384

class EmbeddingCache:
    """Simple in-memory cache for embeddings"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        text_hash = hash(text)
        return self.cache.get(text_hash)

    def set(self, text: str, embedding: List[float]):
        """Cache embedding for text"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        text_hash = hash(text)
        self.cache[text_hash] = embedding

    def clear(self):
        """Clear cache"""
        self.cache.clear()