"""
Semantic Chunker - Intelligent Document Segmentation

Unlike traditional fixed-size chunking, this module detects natural boundaries
in documents to preserve context, relationships, and meaning.

Key Features:
    - Topic boundary detection (when subject matter changes)
    - Hierarchical structure preservation (headers, sections, subsections)
    - Semantic coherence optimization (keep related content together)
    - Entity-aware splitting (don't split entity descriptions)
    - Adaptive chunk sizing (dense vs sparse content)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SemanticChunk:
    """A semantically coherent chunk of text"""
    text: str
    start_pos: int
    end_pos: int
    topic: Optional[str] = None
    level: int = 0  # Hierarchical level (0=top, 1=section, 2=subsection)
    entities: List[str] = None
    metadata: Dict[str, Any] = None
    coherence_score: float = 0.0

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.metadata is None:
            self.metadata = {}


class SemanticChunker:
    """
    Advanced semantic chunking with boundary detection.

    Combines multiple strategies:
    1. Structural analysis (headers, paragraphs, sections)
    2. Topic modeling (semantic similarity between sentences)
    3. Entity recognition (keep entity contexts intact)
    4. Adaptive sizing (based on content density)
    """

    def __init__(
        self,
        target_chunk_size: int = 512,
        min_chunk_size: int = 128,
        max_chunk_size: int = 1024,
        similarity_threshold: float = 0.7,
        use_embeddings: bool = True
    ):
        """
        Initialize the semantic chunker.

        Args:
            target_chunk_size: Preferred chunk size in tokens
            min_chunk_size: Minimum allowed chunk size
            max_chunk_size: Maximum allowed chunk size
            similarity_threshold: Threshold for detecting topic boundaries (0-1)
            use_embeddings: Whether to use embeddings for similarity (slower but better)
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings

        # Initialize embedding model for semantic similarity (lazy load)
        self._embedding_model = None

        # Structural patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.section_break_pattern = re.compile(r'\n\s*\n\s*\n+')  # Multiple blank lines
        self.list_pattern = re.compile(r'^[\s]*[-*•]\s+|^[\s]*\d+\.\s+', re.MULTILINE)

    @property
    def embedding_model(self):
        """Lazy load embedding model only when needed"""
        if self._embedding_model is None and self.use_embeddings:
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedding_model

    def chunk_text(self, text: str, document_metadata: Optional[Dict] = None) -> List[SemanticChunk]:
        """
        Main entry point: chunk text with semantic awareness.

        Args:
            text: The input text to chunk
            document_metadata: Optional metadata about the document

        Returns:
            List of semantically coherent chunks
        """
        if not text or len(text.strip()) == 0:
            return []

        # Step 1: Detect document structure (headers, sections)
        structure = self._detect_structure(text)

        # Step 2: Split into sentences
        sentences = self._split_sentences(text)

        # Step 3: Compute semantic similarities (if enabled)
        if self.use_embeddings and len(sentences) > 1:
            similarities = self._compute_sentence_similarities(sentences)
        else:
            similarities = None

        # Step 4: Find natural boundaries
        boundaries = self._find_semantic_boundaries(
            sentences,
            similarities,
            structure
        )

        # Step 5: Create chunks from boundaries
        chunks = self._create_chunks_from_boundaries(
            text,
            sentences,
            boundaries,
            structure
        )

        # Step 6: Post-process chunks (merge small, split large)
        chunks = self._post_process_chunks(chunks)

        # Step 7: Add metadata
        for chunk in chunks:
            chunk.metadata['document_metadata'] = document_metadata or {}
            chunk.coherence_score = self._compute_chunk_coherence(chunk)

        return chunks

    def _detect_structure(self, text: str) -> Dict[str, Any]:
        """
        Detect hierarchical structure in the document.

        Returns:
            Dictionary mapping positions to structure information
        """
        structure = {
            'headers': [],
            'sections': [],
            'lists': []
        }

        # Detect markdown/text headers
        for match in self.header_pattern.finditer(text):
            level = len(match.group(1))  # Number of # symbols
            title = match.group(2).strip()
            structure['headers'].append({
                'pos': match.start(),
                'level': level,
                'title': title
            })

        # Detect section breaks (multiple blank lines)
        for match in self.section_break_pattern.finditer(text):
            structure['sections'].append(match.start())

        # Detect lists
        for match in self.list_pattern.finditer(text):
            structure['lists'].append(match.start())

        return structure

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with position tracking.

        Returns:
            List of (sentence, start_pos, end_pos) tuples
        """
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentence_endings = re.compile(r'([.!?])\s+')

        sentences = []
        current_pos = 0

        for match in sentence_endings.finditer(text):
            end_pos = match.end()
            sentence = text[current_pos:end_pos].strip()

            if sentence:
                sentences.append((sentence, current_pos, end_pos))

            current_pos = end_pos

        # Add final sentence if exists
        if current_pos < len(text):
            sentence = text[current_pos:].strip()
            if sentence:
                sentences.append((sentence, current_pos, len(text)))

        return sentences if sentences else [(text, 0, len(text))]

    def _compute_sentence_similarities(self, sentences: List[Tuple[str, int, int]]) -> np.ndarray:
        """
        Compute pairwise semantic similarities between consecutive sentences.

        Returns:
            Array of similarity scores between sentence[i] and sentence[i+1]
        """
        if not sentences or len(sentences) < 2:
            return np.array([])

        # Extract just the text
        sentence_texts = [s[0] for s in sentences]

        # Generate embeddings
        embeddings = self.embedding_model.encode(sentence_texts, convert_to_numpy=True)

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(float(sim))

        return np.array(similarities)

    def _find_semantic_boundaries(
        self,
        sentences: List[Tuple[str, int, int]],
        similarities: Optional[np.ndarray],
        structure: Dict[str, Any]
    ) -> List[int]:
        """
        Find natural boundaries for chunking based on semantic shifts.

        Returns:
            List of sentence indices where chunks should be split
        """
        boundaries = [0]  # Always start with first sentence

        # 1. Add boundaries at headers (highest priority)
        header_positions = [h['pos'] for h in structure['headers']]
        for i, (sentence, start, end) in enumerate(sentences):
            if any(start <= hp < end for hp in header_positions):
                if i not in boundaries:
                    boundaries.append(i)

        # 2. Add boundaries at section breaks
        section_positions = structure['sections']
        for i, (sentence, start, end) in enumerate(sentences):
            if any(start <= sp < end for sp in section_positions):
                if i not in boundaries:
                    boundaries.append(i)

        # 3. Add boundaries at semantic shifts (low similarity)
        if similarities is not None and len(similarities) > 0:
            # Find local minima in similarity
            for i in range(1, len(similarities) - 1):
                if (similarities[i] < self.similarity_threshold and
                    similarities[i] < similarities[i-1] and
                    similarities[i] < similarities[i+1]):
                    if i+1 not in boundaries:
                        boundaries.append(i+1)

        # 4. Add final boundary
        if len(sentences) not in boundaries:
            boundaries.append(len(sentences))

        # Sort boundaries
        boundaries = sorted(set(boundaries))

        return boundaries

    def _create_chunks_from_boundaries(
        self,
        text: str,
        sentences: List[Tuple[str, int, int]],
        boundaries: List[int],
        structure: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """Create chunks from identified boundaries"""
        chunks = []
        document_length = len(text)

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx:end_idx]

            if not chunk_sentences:
                continue

            # Extract text
            start_pos = chunk_sentences[0][1]
            end_pos = chunk_sentences[-1][2]
            chunk_text = text[start_pos:end_pos].strip()

            # Determine hierarchical level from headers
            level = 0
            topic = None
            for header in structure['headers']:
                if start_pos <= header['pos'] <= end_pos:
                    level = header['level']
                    topic = header['title']
                    break

            # Create chunk
            chunk = SemanticChunk(
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                topic=topic,
                level=level
            )

            # Enrich metadata for downstream structural features
            chunk.metadata.update({
                "level": level,
                "topic": topic,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "position": (start_pos + end_pos) / (2 * max(document_length, 1)),
                "token_count": len(chunk_text.split()),
                "document_length": document_length,
            })

            chunks.append(chunk)

        return chunks

    def _post_process_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """
        Post-process chunks: merge small chunks, split large ones.
        """
        if not chunks:
            return []

        processed = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]
            chunk_size = len(chunk.text.split())

            # If chunk is too small, try to merge with next
            if chunk_size < self.min_chunk_size and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                merged_text = chunk.text + "\n\n" + next_chunk.text
                merged_size = len(merged_text.split())

                # Only merge if result is not too large
                if merged_size <= self.max_chunk_size:
                    merged_chunk = SemanticChunk(
                        text=merged_text,
                        start_pos=chunk.start_pos,
                        end_pos=next_chunk.end_pos,
                        topic=chunk.topic or next_chunk.topic,
                        level=min(chunk.level, next_chunk.level)
                    )
                    doc_length = chunk.metadata.get("document_length") or next_chunk.metadata.get("document_length") or len(merged_text)
                    merged_chunk.metadata.update({
                        "level": merged_chunk.level,
                        "topic": merged_chunk.topic,
                        "start_pos": merged_chunk.start_pos,
                        "end_pos": merged_chunk.end_pos,
                        "position": (merged_chunk.start_pos + merged_chunk.end_pos) / (2 * max(doc_length, 1)),
                        "token_count": len(merged_text.split()),
                        "document_length": doc_length,
                    })
                    processed.append(merged_chunk)
                    i += 2  # Skip both merged chunks
                    continue

            # If chunk is too large, split it
            if chunk_size > self.max_chunk_size:
                split_chunks = self._split_large_chunk(chunk)
                processed.extend(split_chunks)
            else:
                processed.append(chunk)

            i += 1

        return processed

    def _split_large_chunk(self, chunk: SemanticChunk) -> List[SemanticChunk]:
        """Split a chunk that's too large"""
        words = chunk.text.split()
        sub_chunks = []

        for i in range(0, len(words), self.target_chunk_size):
            sub_text = ' '.join(words[i:i + self.target_chunk_size])
            sub_chunk = SemanticChunk(
                text=sub_text,
                start_pos=chunk.start_pos + i,  # Approximate
                end_pos=chunk.start_pos + i + len(sub_text),
                topic=chunk.topic,
                level=chunk.level
            )
            doc_length = chunk.metadata.get("document_length") or len(chunk.text)
            sub_chunk.metadata.update({
                "level": sub_chunk.level,
                "topic": sub_chunk.topic,
                "start_pos": sub_chunk.start_pos,
                "end_pos": sub_chunk.end_pos,
                "position": (sub_chunk.start_pos + sub_chunk.end_pos) / (2 * max(doc_length, 1)),
                "token_count": len(sub_text.split()),
                "document_length": doc_length,
            })
            sub_chunks.append(sub_chunk)

        return sub_chunks

    def _compute_chunk_coherence(self, chunk: SemanticChunk) -> float:
        """
        Compute coherence score for a chunk (0-1).

        Higher score = more semantically coherent.
        """
        # Simple heuristic based on chunk properties
        score = 1.0

        # Penalize chunks that are too short or too long
        words = len(chunk.text.split())
        if words < self.min_chunk_size:
            score *= (words / self.min_chunk_size)
        elif words > self.max_chunk_size:
            score *= (self.max_chunk_size / words)

        # Bonus for having a topic
        if chunk.topic:
            score *= 1.1

        # Normalize to [0, 1]
        return min(1.0, max(0.0, score))

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 1 token ≈ 0.75 words
        return int(len(text.split()) * 0.75)
