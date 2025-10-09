"""
Query Understanding - Intent Detection and Decomposition

Handles Phase 2 responsibilities for interpreting user queries:
    - Intent detection (question type, complexity scoring)
    - Query decomposition for multi-part questions
    - Query expansion and reformulation hints
    - Lightweight entity extraction for downstream reasoning
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import re


QUESTION_TYPES = {
    "definition": {"what is", "define", "explain"},
    "comparison": {"compare", "difference", "vs", "versus"},
    "procedural": {"how do", "steps", "process", "guide"},
    "diagnostic": {"why does", "cause", "troubleshoot", "issue"},
    "decision": {"should", "best", "recommend", "choose"},
    "factual": {"when", "where", "who", "how many"},
}

DEFAULT_LLM_TEMPERATURE = 0.2
LLM_SYSTEM_PROMPT = (
    "You are an analytical assistant that prepares queries for a retrieval augmented generation system. "
    "Return structured JSON only, no additional text."
)


@dataclass
class QueryIntent:
    """Structured understanding of a query."""

    question_type: str
    intents: List[str]
    complexity: str
    entities: List[str]
    sub_queries: List[str] = field(default_factory=list)
    expansion_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class QueryUnderstandingEngine:
    """Interpret queries to prime retrieval, reasoning, and synthesis phases."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_service: Optional[Any] = None,
        llm_temperature: float = DEFAULT_LLM_TEMPERATURE,
    ) -> None:
        self._entity_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
        self.use_llm = use_llm
        self.llm_service = llm_service
        self.llm_temperature = llm_temperature

    def analyze(self, query: str, conversation_history: Optional[List[str]] = None) -> QueryIntent:
        """
        Analyze a user query.

        Args:
            query: Raw user query text.
            conversation_history: Optional list of recent dialogue turns for context.

        Returns:
            QueryIntent with detected metadata.
        """
        normalized = query.strip().lower()

        question_type = self._detect_question_type(normalized)
        intents = self._detect_intents(normalized)
        if question_type != "general" and question_type not in intents:
            intents.append(question_type)
        complexity = self._estimate_complexity(normalized)
        entities = self._extract_entities(query, conversation_history or [])
        sub_queries = self._decompose_query(query)
        expansion_terms = self._suggest_expansion_terms(normalized, entities)

        metadata = {
            "original_query": query,
            "token_length": str(len(query.split())),
        }

        return QueryIntent(
            question_type=question_type,
            intents=intents,
            complexity=complexity,
            entities=entities,
            sub_queries=sub_queries,
            expansion_terms=expansion_terms,
            metadata=metadata,
        )

    async def analyze_async(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None
    ) -> QueryIntent:
        history = conversation_history or []

        rule_based = self.analyze(query, history)

        if self.use_llm and self.llm_service is not None:
            try:
                return await self._analyze_with_llm(query, history, rule_based)
            except Exception:
                return rule_based

        return rule_based

    def _detect_question_type(self, normalized_query: str) -> str:
        scores = {}
        for question_type, indicators in QUESTION_TYPES.items():
            matches = sum(1 for indicator in indicators if indicator in normalized_query)
            if matches:
                scores[question_type] = matches
        if scores:
            priority = ["comparison", "procedural", "diagnostic", "decision", "definition", "factual"]
            return sorted(
                scores.items(),
                key=lambda item: (
                    -item[1],
                    priority.index(item[0]) if item[0] in priority else len(priority)
                )
            )[0][0]
        if normalized_query.startswith("how "):
            return "procedural"
        return "general"

    def _detect_intents(self, normalized_query: str) -> List[str]:
        intents: Set[str] = set()
        for question_type, indicators in QUESTION_TYPES.items():
            if any(indicator in normalized_query for indicator in indicators):
                intents.add(question_type)

        if " and " in normalized_query or "?" in normalized_query.strip("? "):
            intents.add("multi_part")

        if any(term in normalized_query for term in ["step", "first", "next", "finally"]):
            intents.add("procedural")

        if "why" in normalized_query or "because" in normalized_query:
            intents.add("causal")

        return sorted(intents) if intents else ["general"]

    def _estimate_complexity(self, normalized_query: str) -> str:
        tokens = normalized_query.split()
        if len(tokens) <= 6:
            return "simple"
        if any(term in normalized_query for term in ["compare", "difference", "multiple", "steps", "process"]):
            return "complex"
        if len(tokens) > 20 or " and " in normalized_query or any(op in normalized_query for op in [" vs ", " / ", " or "]):
            return "complex"
        return "moderate"

    def _extract_entities(self, query: str, history: List[str]) -> List[str]:
        candidates = self._scan_entities(query)
        filtered = self._filter_entities(candidates)

        if not filtered and history:
            tail_context = " ".join(history[-2:])
            history_candidates = self._scan_entities(tail_context)
            filtered = self._filter_entities(history_candidates)

        return sorted(filtered)

    def _scan_entities(self, text: str) -> List[str]:
        return [match.group(1).strip() for match in self._entity_pattern.finditer(text)]

    def _filter_entities(self, entities: List[str]) -> List[str]:
        stopwords = {"what", "how", "the", "this", "that", "they", "them", "it", "we"}
        filtered: List[str] = []
        seen = set()
        for entity in entities:
            normalized = entity.lower()
            if len(entity) <= 2 or normalized in stopwords:
                continue
            if normalized in seen:
                continue
            filtered.append(entity)
            seen.add(normalized)
        return filtered

    def _decompose_query(self, query: str) -> List[str]:
        # Split on conjunctions while respecting quoted text
        segments = re.split(r"\band\b|\bor\b|;|,|\?", query)
        sub_queries = [segment.strip() for segment in segments if segment.strip()]
        # Only keep meaningful sub-queries (more than 3 tokens)
        return [sq for sq in sub_queries if len(sq.split()) > 3 and len(sq) <= len(query)]

    def _suggest_expansion_terms(self, normalized_query: str, entities: List[str]) -> List[str]:
        expansions: Set[str] = set()

        if "best" in normalized_query or "recommend" in normalized_query:
            expansions.update({"alternatives", "pros", "cons"})

        if "pricing" in normalized_query or "cost" in normalized_query:
            expansions.update({"tiers", "plans", "discount"})

        if "integration" in normalized_query:
            expansions.update({"api", "authentication", "configuration"})

        for entity in entities:
            if " vs " in normalized_query or "compare" in normalized_query:
                expansions.add(f"{entity} advantages")
                expansions.add(f"{entity} disadvantages")

        return sorted(expansions)

    async def _analyze_with_llm(
        self,
        query: str,
        history: List[str],
        fallback: QueryIntent
    ) -> QueryIntent:

        prompt = self._build_llm_prompt(query, history)
        response = await self.llm_service.generate_response(
            prompt=prompt,
            system_prompt=LLM_SYSTEM_PROMPT,
            temperature=self.llm_temperature,
            max_tokens=600,
        )

        payload = self._parse_llm_response(response)
        if payload is None:
            return fallback

        question_type = self._normalise_question_type(
            payload.get("primary_intent") or payload.get("intent") or fallback.question_type
        )

        llm_intents = self._safe_list(payload.get("sub_intents") or payload.get("intents"))
        intents = self._merge_lists(fallback.intents, [question_type] + llm_intents)

        complexity = payload.get("complexity") or fallback.complexity
        entities = self._merge_lists(
            fallback.entities,
            self._safe_list(payload.get("entities") or payload.get("key_entities"))
        )
        sub_queries = self._merge_lists(
            fallback.sub_queries,
            self._safe_list(payload.get("sub_questions") or payload.get("sub_queries"))
        )
        expansion_terms = self._merge_lists(
            fallback.expansion_terms,
            self._safe_list(payload.get("query_reformulations") or payload.get("expansions"))
        )

        metadata = dict(fallback.metadata)
        if payload.get("implicit_requirements"):
            metadata["implicit_requirements"] = ", ".join(
                self._safe_list(payload["implicit_requirements"])
            )
        if payload.get("ambiguities"):
            metadata["ambiguities"] = ", ".join(self._safe_list(payload["ambiguities"]))
        metadata["intent_source"] = "llm"

        return QueryIntent(
            question_type=question_type,
            intents=intents,
            complexity=complexity,
            entities=entities,
            sub_queries=sub_queries,
            expansion_terms=expansion_terms,
            metadata=metadata,
        )

    def _build_llm_prompt(self, query: str, history: List[str]) -> str:
        history_text = "\n".join(history[-10:]) if history else "No prior conversation."
        return (
            "Analyze the user query and respond with JSON containing the following keys:\n"
            "primary_intent (string from [definition, comparison, procedural, diagnostic, decision, factual]),\n"
            "sub_intents (array of strings),\n"
            "entities (array of strings),\n"
            "implicit_requirements (array of strings),\n"
            "query_reformulations (array of strings),\n"
            "sub_questions (array of strings),\n"
            "ambiguities (array of strings),\n"
            "complexity (string: simple/moderate/complex).\n"
            "Ensure the response is valid JSON only.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Current query: {query}"
        )

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        if not response:
            return None

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return None

        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None

        return None

    def _normalise_question_type(self, label: str) -> str:
        if not label:
            return "general"
        lowered = label.lower()
        for candidate in QUESTION_TYPES.keys():
            if candidate in lowered:
                return candidate
        synonyms = {
            "comparison": {"compare", "versus", "diff"},
            "procedural": {"how-to", "steps", "process"},
            "diagnostic": {"troubleshoot", "why"},
            "decision": {"should", "recommend"},
            "definition": {"definition", "explain"},
            "factual": {"when", "where", "who"},
        }
        for candidate, keys in synonyms.items():
            if any(key in lowered for key in keys):
                return candidate
        return "general"

    def _safe_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if isinstance(item, (str, int, float))]
        return []

    def _merge_lists(self, base: List[str], additions: List[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for item in (base or []) + (additions or []):
            normalized = item.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(normalized)
        return merged
