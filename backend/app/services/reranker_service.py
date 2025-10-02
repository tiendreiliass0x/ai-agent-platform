"""Lightweight cross-encoder reranker for retrieved passages."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Any, Optional


class RerankerService:
    """Wraps an optional cross-encoder model for passage reranking."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._load_error: Optional[str] = None

    async def rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return items sorted by cross-encoder relevance.

        Falls back to the original ordering if the model cannot be loaded or
        either side of the pair is missing text.
        """

        if not items or not query:
            return items

        model = await self._get_model()
        if model is None:
            return items

        pairs = []
        valid_items = []
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            pairs.append([query, text])
            valid_items.append(item)

        if not pairs:
            return items

        loop = asyncio.get_event_loop()

        try:
            scores = await loop.run_in_executor(None, model.predict, pairs)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._load_error = f"Failed to run reranker inference: {exc}"
            return items

        reranked: List[Dict[str, Any]] = []
        for item, score in zip(valid_items, scores):
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            # Blend original similarity score if available for stability
            base_score = float(item.get("score", 0.0) or 0.0)
            enriched["combined_score"] = (enriched["rerank_score"] * 0.7) + (base_score * 0.3)
            reranked.append(enriched)

        reranked.sort(key=lambda x: x.get("combined_score", x.get("rerank_score", 0.0)), reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        # Append any items that were skipped (missing text) in original order
        skipped = [item for item in items if item not in valid_items]
        return reranked + skipped

    async def _get_model(self):
        if self._model is not None or self._load_error is not None:
            return self._model

        loop = asyncio.get_event_loop()
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            def _load():
                return CrossEncoder(self.model_name)

            self._model = await loop.run_in_executor(None, _load)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._load_error = f"Unable to load reranker model '{self.model_name}': {exc}"
            self._model = None

        return self._model


reranker_service = RerankerService()
