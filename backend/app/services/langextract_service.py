"""LangExtract integration with graceful fallback."""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import settings

try:  # pragma: no cover
    from langextract import LangExtract
except ImportError:  # pragma: no cover
    LangExtract = None  # type: ignore


@dataclass
class LangExtractResult:
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]


class LangExtractService:
    """Thin wrapper around LangExtract with a simple heuristic fallback."""

    def __init__(self) -> None:
        self.api_key = settings.LANGEXTRACT_API_KEY
        self._client = None

        if LangExtract and self.api_key:
            try:  # pragma: no cover - rely on runtime availability
                self._client = LangExtract(api_key=self.api_key)
            except Exception as exc:
                # Fallback to heuristic if sdk init fails
                print(f"[LangExtract] Initialization failed: {exc}")
                self._client = None

        # simple sentiment heuristic fallback vocabulary
        self._positive_words = {
            "awesome",
            "great",
            "excellent",
            "fast",
            "reliable",
            "love",
            "good",
            "smooth",
            "efficient",
            "easy",
        }
        self._negative_words = {
            "bad",
            "slow",
            "problem",
            "issue",
            "hate",
            "difficult",
            "error",
            "broken",
            "delay",
            "confused",
        }

    async def analyze_text(self, text: str, language: Optional[str] = None) -> Optional[LangExtractResult]:
        if not text or not text.strip():
            return None

        if self._client:
            loop = asyncio.get_event_loop()
            try:  # pragma: no cover - dependent on external SDK
                result = await loop.run_in_executor(
                    None,
                    lambda: self._client.analyze(
                        text=text,
                        language=language,
                        features=["sentiment", "entities"],
                    ),
                )
                sentiment = result.get("sentiment", {})
                entities = result.get("entities", [])
                return LangExtractResult(sentiment=sentiment, entities=entities)
            except Exception as exc:
                print(f"[LangExtract] analyze() failed, falling back to heuristics: {exc}")

        return self._heuristic_analysis(text)

    def _heuristic_analysis(self, text: str) -> LangExtractResult:
        """Lightweight fallback sentiment + entity detection."""
        words = re.findall(r"[\w']+", text.lower())
        pos_hits = sum(1 for w in words if w in self._positive_words)
        neg_hits = sum(1 for w in words if w in self._negative_words)
        total = pos_hits + neg_hits

        if total == 0:
            sentiment = {"label": "neutral", "score": 0.0}
        elif pos_hits > neg_hits:
            sentiment = {
                "label": "positive",
                "score": round(pos_hits / total, 3),
            }
        elif neg_hits > pos_hits:
            sentiment = {
                "label": "negative",
                "score": round(neg_hits / total, 3),
            }
        else:
            sentiment = {"label": "neutral", "score": 0.0}

        # Simple entity heuristic: consecutive capitalized words
        entities: List[Dict[str, Any]] = []
        for match in re.finditer(r"((?:[A-Z][a-z]+\s?){1,3})", text):
            name = match.group(1).strip()
            if len(name.split()) == 1 and len(name) < 3:
                continue
            entities.append({
                "text": name,
                "type": "PROPER_NOUN",
                "start": match.start(),
                "end": match.end(),
            })

        return LangExtractResult(sentiment=sentiment, entities=entities)


# Shared instance
lang_extract_service = LangExtractService()
