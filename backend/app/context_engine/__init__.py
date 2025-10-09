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
    7. Query Understanding - Intent detection and decomposition
    8. Conversation Memory - Multi-turn context tracking
    9. Multi-Hop Reasoning - Evidence stitching
    10. Confidence Scoring - Answer reliability
    11. Contradiction Detection - Cross-source conflict checks
    12. Answer Synthesis - Multi-source fusion

This is the moat that makes agents irreplaceable.
"""

from .semantic_chunker import SemanticChunker
from .multi_vector_embedder import MultiVectorEmbedder
from .hybrid_retriever import HybridRetriever
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .source_tracker import SourceTracker
from .quality_metrics import ContextQualityMetrics
from .reranker import Reranker
from .query_understanding import QueryUnderstandingEngine
from .conversation_memory import ConversationMemory
from .conversation_store import ConversationMemoryStore
from .multi_hop_reasoner import MultiHopReasoner
from .confidence_scorer import ConfidenceScorer
from .contradiction_detector import ContradictionDetector
from .answer_synthesizer import AnswerSynthesizer

__all__ = [
    "SemanticChunker",
    "MultiVectorEmbedder",
    "HybridRetriever",
    "KnowledgeGraphBuilder",
    "SourceTracker",
    "ContextQualityMetrics",
    "Reranker",
    "QueryUnderstandingEngine",
    "ConversationMemory",
    "ConversationMemoryStore",
    "MultiHopReasoner",
    "ConfidenceScorer",
    "ContradictionDetector",
    "AnswerSynthesizer",
]

__version__ = "0.2.0"
