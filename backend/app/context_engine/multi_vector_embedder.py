"""
Multi-Vector Embeddings - Rich Representation System

Generates multiple complementary embeddings for each content piece:
1. Semantic Embeddings: Dense vectors capturing meaning
2. Keyword Embeddings: TF-IDF sparse vectors for keyword matching
3. Structural Embeddings: Position and hierarchy information

Why Multi-Vector?
- Semantic: Captures "AI agents" ≈ "intelligent systems"
- Keyword: Ensures exact matches for "pricing", "API key"
- Structural: Prioritizes headings, first paragraphs, summaries
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import math


@dataclass
class MultiVectorEmbedding:
    """Multi-vector representation of a text chunk"""
    text: str
    chunk_id: str

    # Semantic vector (dense)
    semantic_embedding: Optional[np.ndarray] = None

    # Keyword vector (sparse TF-IDF)
    keyword_embedding: Optional[np.ndarray] = None

    # Structural features
    structural_features: Dict[str, float] = field(default_factory=dict)
    structural_embedding: Optional[np.ndarray] = None
    structural_feature_names: List[str] = field(default_factory=list)

    # Combined representation
    combined_embedding: Optional[np.ndarray] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "semantic_embedding": self.semantic_embedding.tolist() if self.semantic_embedding is not None else None,
            "keyword_embedding": self.keyword_embedding.tolist() if self.keyword_embedding is not None else None,
            "structural_features": self.structural_features,
            "structural_embedding": self.structural_embedding.tolist() if self.structural_embedding is not None else None,
            "structural_feature_names": self.structural_feature_names,
            "metadata": self.metadata
        }


class MultiVectorEmbedder:
    """
    Generate multiple complementary embeddings for rich retrieval.

    Strategy:
    1. Semantic embeddings via sentence transformers (or provided embedder)
    2. Keyword embeddings via TF-IDF
    3. Structural features (position, level, topic importance)
    4. Fusion: Combine all representations for comprehensive search
    """

    def __init__(
        self,
        semantic_model: Optional[str] = "all-MiniLM-L6-v2",
        max_tfidf_features: int = 5000,
        structural_weight: float = 0.1,
        use_semantic: bool = True,
        use_keyword: bool = True,
        use_structural: bool = True
    ):
        """
        Initialize multi-vector embedder.

        Args:
            semantic_model: Name of sentence transformer model
            max_tfidf_features: Maximum TF-IDF vocabulary size
            structural_weight: Weight for structural features in combined embedding
            use_semantic: Generate semantic embeddings
            use_keyword: Generate keyword embeddings
            use_structural: Generate structural features
        """
        self.semantic_model_name = semantic_model
        self._semantic_model = None  # Lazy load

        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vectorizer = None

        self.structural_weight = structural_weight
        self.use_semantic = use_semantic
        self.use_keyword = use_keyword
        self.use_structural = use_structural

    @property
    def semantic_model(self):
        """Lazy load semantic embedding model"""
        if self._semantic_model is None and self.use_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                self._semantic_model = SentenceTransformer(self.semantic_model_name)
            except ImportError:
                print("Warning: sentence-transformers not installed. Semantic embeddings disabled.")
                self.use_semantic = False
        return self._semantic_model

    def fit_keyword_embedder(self, corpus: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on corpus.

        Args:
            corpus: List of text documents
        """
        if not self.use_keyword or len(corpus) == 0:
            return

        # Adjust max_df for small corpora
        max_df = 0.95 if len(corpus) > 5 else 1.0

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,  # Minimum document frequency
            max_df=max_df  # Adjust for corpus size
        )

        # Fit on corpus
        self.tfidf_vectorizer.fit(corpus)

    def generate_semantic_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate semantic embedding using sentence transformer.

        Args:
            text: Input text

        Returns:
            Semantic embedding vector
        """
        if not self.use_semantic or self.semantic_model is None:
            return None

        embedding = self.semantic_model.encode(text, convert_to_numpy=True)
        return embedding

    def generate_keyword_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate keyword embedding using TF-IDF.

        Args:
            text: Input text

        Returns:
            TF-IDF vector (sparse -> dense)
        """
        if not self.use_keyword or self.tfidf_vectorizer is None:
            return None

        # Transform text to TF-IDF vector
        tfidf_vector = self.tfidf_vectorizer.transform([text])

        # Convert sparse to dense
        dense_vector = tfidf_vector.toarray()[0]

        return dense_vector

    def generate_structural_features(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Generate structural features for content positioning.

        Features:
        - Level: Hierarchical depth (0 = top level, 1 = subsection, etc.)
        - Position: Normalized position in document (0 = start, 1 = end)
        - Length: Normalized chunk length
        - Topic importance: Is this a header/title?
        - Keyword density: Ratio of important keywords

        Args:
            text: Input text
            metadata: Optional metadata with structural info

        Returns:
            Dictionary of structural features
        """
        if not self.use_structural:
            return {}

        features = {}
        metadata = metadata or {}

        # Level (hierarchical depth)
        features['level'] = float(metadata.get('level', 0))

        # Position in document (0-1)
        features['position'] = float(metadata.get('position', 0.5))

        # Normalized length (log scale to prevent dominance)
        text_length = len(text.split())
        features['length'] = math.log(text_length + 1) / math.log(1000)  # Normalized to ~1000 words

        # Topic importance (is this a header/title?)
        features['is_topic'] = 1.0 if metadata.get('topic') else 0.0

        # Keyword density (simple heuristic)
        words = text.lower().split()
        word_counts = Counter(words)
        unique_ratio = len(word_counts) / max(len(words), 1)
        features['keyword_density'] = unique_ratio

        # Start bias (content at start is often more important)
        features['start_bias'] = 1.0 - features['position']

        return features

    def combine_embeddings(
        self,
        semantic_emb: Optional[np.ndarray],
        keyword_emb: Optional[np.ndarray],
        structural_features: Dict[str, float]
    ) -> Optional[np.ndarray]:
        """
        Combine multiple embeddings into unified representation.

        Strategy:
        - Concatenate semantic + keyword embeddings
        - Append structural features as additional dimensions
        - Normalize to unit length

        Args:
            semantic_emb: Semantic embedding vector
            keyword_emb: Keyword embedding vector
            structural_features: Structural feature dict

        Returns:
            Combined embedding vector
        """
        vectors = []

        if semantic_emb is not None:
            vectors.append(semantic_emb)

        if keyword_emb is not None:
            vectors.append(keyword_emb)

        if structural_features and self.use_structural:
            # Convert structural features to vector
            struct_vector = np.array(
                [structural_features[key] for key in sorted(structural_features.keys())],
                dtype=float
            )
            # Scale structural features by weight
            struct_vector = struct_vector * self.structural_weight
            vectors.append(struct_vector)

        if not vectors:
            return None

        # Concatenate all vectors
        combined = np.concatenate(vectors)

        # Normalize to unit length
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def embed_chunk(
        self,
        text: str,
        chunk_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MultiVectorEmbedding:
        """
        Generate multi-vector embedding for a single chunk.

        Args:
            text: Text content
            chunk_id: Unique chunk identifier
            metadata: Optional metadata

        Returns:
            MultiVectorEmbedding object
        """
        metadata = metadata or {}

        # 1. Generate semantic embedding
        semantic_emb = None
        if self.use_semantic:
            semantic_emb = self.generate_semantic_embedding(text)

        # 2. Generate keyword embedding
        keyword_emb = None
        if self.use_keyword:
            keyword_emb = self.generate_keyword_embedding(text)

        # 3. Generate structural features
        structural_features = {}
        structural_embedding = None
        structural_feature_names: List[str] = []
        if self.use_structural:
            structural_features = self.generate_structural_features(text, metadata)
            if structural_features:
                structural_feature_names = sorted(structural_features.keys())
                structural_embedding = np.array(
                    [structural_features[name] for name in structural_feature_names],
                    dtype=float
                )

        # 4. Combine embeddings
        combined_emb = self.combine_embeddings(
            semantic_emb, keyword_emb, structural_features
        )

        return MultiVectorEmbedding(
            text=text,
            chunk_id=chunk_id,
            semantic_embedding=semantic_emb,
            keyword_embedding=keyword_emb,
            structural_features=structural_features,
            structural_embedding=structural_embedding,
            structural_feature_names=structural_feature_names,
            combined_embedding=combined_emb,
            metadata=metadata
        )

    def embed_chunks(
        self,
        chunks: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> List[MultiVectorEmbedding]:
        """
        Generate multi-vector embeddings for multiple chunks.

        Args:
            chunks: List of (text, chunk_id, metadata) tuples

        Returns:
            List of MultiVectorEmbedding objects
        """
        # First pass: collect all text for TF-IDF fitting
        if self.use_keyword and self.tfidf_vectorizer is None:
            corpus = [text for text, _, _ in chunks]
            self.fit_keyword_embedder(corpus)

        # Second pass: embed each chunk
        embeddings = []
        for text, chunk_id, metadata in chunks:
            embedding = self.embed_chunk(text, chunk_id, metadata)
            embeddings.append(embedding)

        return embeddings

    def embed_document(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Backwards-compatible helper that returns numpy arrays for pipeline use.

        Args:
            text: Document or chunk text
            doc_id: Identifier for the resulting embedding
            metadata: Optional metadata carried through the pipeline

        Returns:
            Dictionary with semantic, keyword, structural, and combined embeddings
        """
        embedding = self.embed_chunk(text, doc_id, metadata)

        return {
            "semantic": embedding.semantic_embedding,
            "keyword": embedding.keyword_embedding,
            "structural": embedding.structural_embedding,
            "combined": embedding.combined_embedding,
            "metadata": embedding.metadata,
            "structural_feature_names": embedding.structural_feature_names,
        }

    def embed_query(
        self,
        query: str,
        boost_keywords: bool = True
    ) -> MultiVectorEmbedding:
        """
        Generate multi-vector embedding for a query.

        Args:
            query: Search query
            boost_keywords: Increase keyword weight for queries

        Returns:
            MultiVectorEmbedding for the query
        """
        # Queries don't need structural features
        original_structural = self.use_structural
        self.use_structural = False

        embedding = self.embed_chunk(
            text=query,
            chunk_id="query",
            metadata={"is_query": True}
        )

        self.use_structural = original_structural

        return embedding

    def similarity_search(
        self,
        query_embedding: MultiVectorEmbedding,
        chunk_embeddings: List[MultiVectorEmbedding],
        top_k: int = 10,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Tuple[str, float]]:
        """
        Search using multi-vector similarity.

        Args:
            query_embedding: Query embedding
            chunk_embeddings: Chunk embeddings to search
            top_k: Number of results
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword similarity

        Returns:
            List of (chunk_id, score) tuples
        """
        scores = []

        for chunk_emb in chunk_embeddings:
            score = 0.0

            # Semantic similarity
            if query_embedding.semantic_embedding is not None and chunk_emb.semantic_embedding is not None:
                q_norm = np.linalg.norm(query_embedding.semantic_embedding)
                c_norm = np.linalg.norm(chunk_emb.semantic_embedding)

                if q_norm > 0 and c_norm > 0:
                    semantic_sim = np.dot(
                        query_embedding.semantic_embedding,
                        chunk_emb.semantic_embedding
                    ) / (q_norm * c_norm)
                    score += semantic_weight * semantic_sim

            # Keyword similarity
            if query_embedding.keyword_embedding is not None and chunk_emb.keyword_embedding is not None:
                q_norm = np.linalg.norm(query_embedding.keyword_embedding)
                c_norm = np.linalg.norm(chunk_emb.keyword_embedding)

                if q_norm > 0 and c_norm > 0:
                    keyword_sim = np.dot(
                        query_embedding.keyword_embedding,
                        chunk_emb.keyword_embedding
                    ) / (q_norm * c_norm)
                    score += keyword_weight * keyword_sim

            scores.append((chunk_emb.chunk_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
