"""
Lightweight text processing utilities - Langchain replacement
"""
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class SimpleDocument:
    """Simple document class to replace langchain.docstore.document.Document"""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleTextSplitter:
    """Lightweight text splitter to replace langchain RecursiveCharacterTextSplitter"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []

        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""

                for part in parts:
                    # If adding this part would exceed chunk size, save current chunk
                    if len(current_chunk) + len(part) + len(separator) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())

                            # Start new chunk with overlap from previous chunk
                            if self.chunk_overlap > 0:
                                overlap_text = current_chunk[-self.chunk_overlap:]
                                current_chunk = overlap_text + separator + part
                            else:
                                current_chunk = part
                        else:
                            # Part itself is too large, recursively split it
                            if len(part) > self.chunk_size:
                                sub_chunks = self._split_large_text(part)
                                chunks.extend(sub_chunks[:-1])  # Add all but last
                                current_chunk = sub_chunks[-1]  # Last becomes current
                            else:
                                current_chunk = part
                    else:
                        # Add part to current chunk
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part

                # Don't forget the last chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())

                return [chunk for chunk in chunks if chunk.strip()]

        # If no separators worked, split by character count
        return self._split_large_text(text)

    def _split_large_text(self, text: str) -> List[str]:
        """Split large text by character count when no separators work"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to break at word boundary
            chunk = text[start:end]
            last_space = chunk.rfind(' ')

            if last_space != -1 and last_space > start + (self.chunk_size * 0.8):
                end = start + last_space

            chunks.append(text[start:end])

            # Move start with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks