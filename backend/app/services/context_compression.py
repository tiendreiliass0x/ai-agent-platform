"""Lightweight contextual compression utilities."""

import re
from typing import Dict, Optional


def compress_context_snippet(
    text: str,
    metadata: Optional[Dict[str, any]] = None,
    max_sentences: int = 3,
    max_chars: int = 600
) -> str:
    """Return a compressed version of the snippet using stored summaries/keywords."""
    if not text:
        return ""

    metadata = metadata or {}

    if metadata.get("focus_summary"):
        summary = metadata["focus_summary"].strip()
        if summary:
            return summary[:max_chars]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text[:max_chars]

    keywords = [kw.lower() for kw in (metadata.get("keywords") or [])]
    prioritized = []
    if keywords:
        for sentence in sentences:
            lowered = sentence.lower()
            if any(kw in lowered for kw in keywords):
                prioritized.append(sentence)
            if len(prioritized) >= max_sentences:
                break

    if not prioritized:
        prioritized = sentences[:max_sentences]

    compressed = " ".join(prioritized)
    return compressed[:max_chars]
