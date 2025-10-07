"""
Context Engine - World-Class Knowledge Architecture for AI Agents

The context engine transforms scattered, messy data into structured, intelligent
knowledge that makes agents genuinely expert-level with nuanced understanding.

Architecture Layers:
    1. Semantic Chunking - Intelligent boundary detection
    2. Multi-Vector Embeddings - Multiple representation aspects
    3. Hybrid Retrieval - Dense + sparse (BM25) search
    4. Knowledge Graph - Entity extraction and relationships
    5. Source Attribution - Provenance tracking with confidence
    6. Quality Metrics - Measurable context excellence

This is the moat that makes agents irreplaceable.
"""

from .semantic_chunker import SemanticChunker
from .multi_vector_embedder import MultiVectorEmbedder
from .hybrid_retriever import HybridRetriever
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .source_tracker import SourceTracker
from .quality_metrics import ContextQualityMetrics
from .reranker import Reranker

__all__ = [
    "SemanticChunker",
    "MultiVectorEmbedder",
    "HybridRetriever",
    "KnowledgeGraphBuilder",
    "SourceTracker",
    "ContextQualityMetrics",
    "Reranker",
]

__version__ = "0.1.0"
