"""
Context Engine Celery Tasks

Heavy processing operations that should run asynchronously:
- Document embedding generation
- Knowledge graph construction
- Large-scale indexing
"""

import os

# macOS + PyTorch (MPS) dislikes forked workers; disable fork safety when needed.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from celery import Task
from app.celery_app import celery_app
from app.context_engine import (
    SemanticChunker,
    MultiVectorEmbedder,
    KnowledgeGraphBuilder,
    HybridRetriever,
    SourceTracker
)
from app.context_engine.source_tracker import AuthorityLevel
import numpy as np
from typing import List, Dict, Any, Optional
import time


class CallbackTask(Task):
    """Task with callbacks for progress tracking"""

    def on_success(self, retval, task_id, args, kwargs):
        """Log success"""
        print(f"✓ Task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log failure"""
        print(f"✗ Task {task_id} failed: {exc}")


@celery_app.task(base=CallbackTask, bind=True)
def process_document_for_context_engine(
    self,
    document_id: int,
    text: str,
    source_metadata: Dict[str, Any],
    agent_id: Optional[int] = None,
    store_in_vector_db: bool = False
) -> Dict[str, Any]:
    """
    Process a document through the full context engine pipeline.

    This runs asynchronously and can take 10-60 seconds for large documents.

    Steps:
    1. Semantic chunking
    2. Multi-vector embedding generation
    3. Knowledge graph construction
    4. Index for hybrid retrieval
    5. Source attribution

    Args:
        document_id: Database document ID
        text: Document text content
        source_metadata: Source info (title, author, url, etc.)

    Returns:
        Processing results with chunk_ids, embeddings, entities
    """
    try:
        # Update task state
        self.update_state(state='PROCESSING', meta={'step': 'chunking', 'progress': 0})

        # Step 1: Semantic Chunking
        print(f"📄 Processing document {document_id}: Chunking...")
        chunker = SemanticChunker(
            target_chunk_size=512,
            min_chunk_size=128,
            max_chunk_size=1024,
            similarity_threshold=0.7
        )
        chunks = chunker.chunk_text(text)
        print(f"✓ Created {len(chunks)} chunks")

        self.update_state(state='PROCESSING', meta={'step': 'embeddings', 'progress': 20})

        # Step 2: Multi-Vector Embeddings (SLOW - why it's in task queue)
        print(f"🎯 Generating embeddings for {len(chunks)} chunks...")
        embedder = MultiVectorEmbedder(
            semantic_model="all-MiniLM-L6-v2",
            use_semantic=True,
            use_keyword=True,
            use_structural=True
        )

        chunk_data = []
        for i, chunk in enumerate(chunks):
            # Update progress
            progress = 20 + int((i / len(chunks)) * 40)
            self.update_state(
                state='PROCESSING',
                meta={'step': 'embeddings', 'progress': progress, 'chunk': f"{i+1}/{len(chunks)}"}
            )

            # Generate embeddings
            embeddings = embedder.embed_document(chunk.text, doc_id=f"{document_id}_chunk_{i}")

            chunk_data.append({
                'chunk_id': f"{document_id}_chunk_{i}",
                'text': chunk.text,
                'start_pos': chunk.start_pos,
                'end_pos': chunk.end_pos,
                'semantic_embedding': embeddings['semantic'].tolist(),
                'keyword_embedding': embeddings['keyword'].tolist() if embeddings['keyword'] is not None else None,
                'structural_features': embeddings['structural'].tolist() if embeddings['structural'] is not None else None,
                'metadata': chunk.metadata
            })

        print(f"✓ Generated embeddings for {len(chunk_data)} chunks")

        self.update_state(state='PROCESSING', meta={'step': 'knowledge_graph', 'progress': 60})

        # Step 3: Knowledge Graph Construction (SLOW - NER + relationship extraction)
        print(f"🕸️  Building knowledge graph...")
        kg_builder = KnowledgeGraphBuilder()

        for data in chunk_data:
            kg_builder.add_document(data['chunk_id'], data['text'])

        entities = [
            {
                'id': entity.id,
                'name': entity.name,
                'text': entity.name,
                'type': entity.type,
                'mentions': entity.mentions
            }
            for entity in kg_builder.entities.values()
        ]

        relationships = [
            {
                'source': rel.source_id,
                'target': rel.target_id,
                'type': rel.relation_type,
                'confidence': rel.confidence
            }
            for rel in kg_builder.relationships
        ]

        print(f"✓ Extracted {len(entities)} entities and {len(relationships)} relationships")

        self.update_state(state='PROCESSING', meta={'step': 'source_attribution', 'progress': 80})

        # Step 4: Source Attribution
        print(f"📚 Creating source attribution...")
        source_tracker = SourceTracker()

        source = source_tracker.register_source(
            source_id=f"doc_{document_id}",
            source_type=source_metadata.get('type', 'document'),
            authority_level=AuthorityLevel[source_metadata.get('authority', 'COMMUNITY').upper()],
            source_url=source_metadata.get('url'),
            metadata=source_metadata
        )

        # Create attributions for each chunk
        attributions = []
        for i, data in enumerate(chunk_data):
            attribution = source_tracker.create_attribution(
                source_id=f"doc_{document_id}",
                confidence=0.95,
                section=f"Chunk {i+1}"
            )
            source_tracker.link_content_to_source(data['chunk_id'], attribution)
            attributions.append({
                'chunk_id': data['chunk_id'],
                'source_id': f"doc_{document_id}",
                'confidence': 0.95
            })

        print(f"✓ Created source attributions")

        # Step 5: (Optional) Store vectors in Pinecone
        vector_ids = []
        if store_in_vector_db and agent_id is not None:
            self.update_state(state='PROCESSING', meta={'step': 'storing_vectors', 'progress': 85})

            from app.services.vector_store import VectorStoreService
            import asyncio

            print(f"📤 Storing {len(chunk_data)} vectors in Pinecone...")

            # Extract semantic embeddings
            embeddings = [c['semantic_embedding'] for c in chunk_data]
            texts = [c['text'] for c in chunk_data]

            # Prepare metadata
            metadatas = []
            for c in chunk_data:
                meta = {
                    'chunk_id': c['chunk_id'],
                    'document_id': document_id,
                    'agent_id': agent_id,
                    'start_pos': c.get('start_pos', 0),
                    'end_pos': c.get('end_pos', 0)
                }
                if c.get('metadata'):
                    meta.update(c['metadata'])
                metadatas.append(meta)

            # Store vectors
            vector_store = VectorStoreService()
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            vector_ids = loop.run_until_complete(
                vector_store.add_vectors(
                    embeddings=embeddings,
                    texts=texts,
                    metadatas=metadatas
                )
            )
            print(f"✓ Stored {len(vector_ids)} vectors in Pinecone")

        self.update_state(state='PROCESSING', meta={'step': 'finalizing', 'progress': 95})

        # Step 6: Return results for storage
        result = {
            'document_id': document_id,
            'chunks': chunk_data,
            'entities': entities,
            'relationships': relationships,
            'attributions': attributions,
            'source_info': {
                'source_id': f"doc_{document_id}",
                'authority_level': source.authority_level.value,
                'timestamp': source.timestamp.isoformat()
            },
            'vector_ids': vector_ids,
            'stats': {
                'total_chunks': len(chunk_data),
                'total_entities': len(entities),
                'total_relationships': len(relationships),
                'total_vectors_stored': len(vector_ids),
                'processing_time': time.time()
            }
        }

        print(f"✅ Document {document_id} processing complete!")

        return result

    except Exception as e:
        print(f"❌ Error processing document {document_id}: {e}")
        raise


@celery_app.task(bind=True)
def generate_embeddings_batch(
    self,
    texts: List[str],
    batch_id: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Generate embeddings for a batch of texts.

    Useful for:
    - Batch processing multiple documents
    - Re-embedding after model updates
    - Query embedding generation (though this should be fast enough to be sync)

    Args:
        texts: List of text chunks to embed
        batch_id: Identifier for this batch
        model_name: Embedding model to use

    Returns:
        Embeddings and metadata
    """
    try:
        self.update_state(state='PROCESSING', meta={'progress': 0, 'total': len(texts)})

        embedder = MultiVectorEmbedder(semantic_model=model_name)

        embeddings = []
        for i, text in enumerate(texts):
            progress = int((i / len(texts)) * 100)
            self.update_state(
                state='PROCESSING',
                meta={'progress': progress, 'current': i+1, 'total': len(texts)}
            )

            emb = embedder.embed_document(text, doc_id=f"{batch_id}_{i}")
            embeddings.append({
                'text_index': i,
                'semantic': emb['semantic'].tolist(),
                'keyword': emb['keyword'].tolist() if emb['keyword'] is not None else None,
                'structural': emb['structural'].tolist() if emb['structural'] is not None else None
            })

        return {
            'batch_id': batch_id,
            'embeddings': embeddings,
            'model': model_name,
            'count': len(embeddings)
        }

    except Exception as e:
        print(f"Error generating embeddings for batch {batch_id}: {e}")
        raise


@celery_app.task(bind=True)
def build_knowledge_graph_from_documents(
    self,
    document_chunks: List[Dict[str, str]],
    graph_id: str
) -> Dict[str, Any]:
    """
    Build knowledge graph from multiple documents.

    This can be VERY slow for large document sets.

    Args:
        document_chunks: List of {'chunk_id': ..., 'text': ...}
        graph_id: Identifier for this graph

    Returns:
        Graph structure with entities and relationships
    """
    try:
        total = len(document_chunks)
        self.update_state(state='PROCESSING', meta={'progress': 0, 'total': total})

        kg_builder = KnowledgeGraphBuilder()

        for i, chunk in enumerate(document_chunks):
            progress = int((i / total) * 100)
            self.update_state(
                state='PROCESSING',
                meta={'progress': progress, 'current': i+1, 'total': total}
            )

            kg_builder.add_document(chunk['chunk_id'], chunk['text'])

        # Extract results
        entities = [
            {
                'id': e.id,
                'text': e.text,
                'type': e.entity_type,
                'mentions': e.mentions
            }
            for e in kg_builder.entities.values()
        ]

        relationships = [
            {
                'source': r.source_id,
                'target': r.target_id,
                'type': r.relation_type,
                'confidence': r.confidence
            }
            for r in kg_builder.relationships
        ]

        return {
            'graph_id': graph_id,
            'entities': entities,
            'relationships': relationships,
            'stats': {
                'total_entities': len(entities),
                'total_relationships': len(relationships),
                'chunks_processed': total
            }
        }

    except Exception as e:
        print(f"Error building knowledge graph {graph_id}: {e}")
        raise


@celery_app.task
def reindex_document_for_hybrid_retrieval(
    document_id: int,
    agent_id: int,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> Dict[str, Any]:
    """
    Index document chunks for hybrid retrieval and store in vector database.

    This prepares data structures for:
    - BM25 sparse retrieval (in-memory for HybridRetriever)
    - Dense vector retrieval (stored in Pinecone)
    - Reciprocal Rank Fusion

    Args:
        document_id: Document ID for tracking
        agent_id: Agent ID for vector store filtering
        chunks: List of chunk dictionaries with text and metadata
        embeddings: List of embedding vectors (as lists, not numpy arrays)

    Returns:
        Indexing results with vector IDs and status
    """
    try:
        from app.services.vector_store import VectorStoreService
        import asyncio

        corpus = [chunk['text'] for chunk in chunks]

        # Prepare metadata for vector store
        metadatas = []
        for chunk in chunks:
            meta = {
                'chunk_id': chunk['chunk_id'],
                'document_id': document_id,
                'agent_id': agent_id,
                'start_pos': chunk.get('start_pos', 0),
                'end_pos': chunk.get('end_pos', 0)
            }
            # Include any additional metadata from chunks
            if 'metadata' in chunk:
                meta.update(chunk['metadata'])
            metadatas.append(meta)

        # Build HybridRetriever for BM25 indexing (in-memory)
        retriever = HybridRetriever(
            corpus=corpus,
            metadata=metadatas,
            embeddings=np.array(embeddings) if embeddings else None
        )

        # Store vectors in Pinecone
        vector_ids = []
        if embeddings:
            print(f"📤 Storing {len(embeddings)} vectors in Pinecone...")
            vector_store = VectorStoreService()

            # Run async operation in sync context (Celery task)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            vector_ids = loop.run_until_complete(
                vector_store.add_vectors(
                    embeddings=embeddings,
                    texts=corpus,
                    metadatas=metadatas
                )
            )
            print(f"✓ Stored {len(vector_ids)} vectors in Pinecone")

        return {
            'document_id': document_id,
            'agent_id': agent_id,
            'indexed_chunks': len(corpus),
            'has_embeddings': embeddings is not None,
            'vector_ids': vector_ids,
            'status': 'indexed'
        }

    except Exception as e:
        print(f"Error indexing document {document_id}: {e}")
        raise
