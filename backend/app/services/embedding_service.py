import asyncio
from typing import List, Any
import openai
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = "text-embedding-3-large"
        self.embedding_dimension = 3072

        # Fallback to sentence-transformers if OpenAI is not available
        self.fallback_model = None

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            if settings.OPENAI_API_KEY:
                return await self._generate_openai_embeddings(texts)
            else:
                return await self._generate_local_embeddings(texts)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Fallback to local model
            return await self._generate_local_embeddings(texts)

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

    def get(self, text: str) -> List[float] | None:
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