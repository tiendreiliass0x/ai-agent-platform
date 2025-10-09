"""
Contradiction Detection - Semantic Conflict Resolution with LLM

Detects conflicting statements across sources using:
    - LLM semantic contradiction detection (deep understanding)
    - Polarity-based heuristics (fallback)
    - Authority levels to guide resolution
    - Temporal conflicts (outdated vs current)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..services.llm_service_interface import LLMServiceInterface


@dataclass
class Statement:
    text: str
    source_id: str
    authority: str | float  # Can be string or numeric
    timestamp: Optional[str] = None
    polarity: Optional[int] = None  # +1 affirmative, -1 negative, 0 neutral


@dataclass
class ContradictionFinding:
    statements: Tuple[Statement, Statement]
    severity: str
    description: str
    rationale: Optional[str] = None  # Alias for backward compatibility
    resolution: Optional[str] = None

    def __post_init__(self):
        # Ensure rationale and description are synced
        if self.rationale is None and self.description:
            self.rationale = self.description
        elif self.description is None and self.rationale:
            self.description = self.rationale


class ContradictionDetector:
    """Contradiction detector with LLM semantic understanding + heuristic fallback."""

    NEGATION_MARKERS = {"not", "never", "cannot", "no longer", "deprecated"}
    AUTHORITY_ORDER = {"official": 3, "verified": 2, "community": 1, "unverified": 0}

    def __init__(
        self,
        llm_service: Optional["LLMServiceInterface"] = None,
        use_llm: bool = True,
        embedder = None
    ) -> None:
        """
        Initialize contradiction detector.

        Args:
            llm_service: LLM service for semantic contradiction detection
            use_llm: Whether to use LLM (True) or fallback to heuristics (False)
            embedder: Embedding service for similarity filtering (optional)
        """
        self.llm_service = llm_service
        self.use_llm = use_llm and llm_service is not None
        self.embedder = embedder

    def detect_conflicts(self, statements: List[Statement]) -> List[ContradictionFinding]:
        """
        Unified contradiction detection (synchronous wrapper).

        Uses heuristic-based detection since this is synchronous.
        For LLM detection, use detect_with_llm_async().

        Args:
            statements: List of statements to check for contradictions

        Returns:
            List of contradiction findings
        """
        return self.detect(statements)

    def detect(self, statements: List[Statement]) -> List[ContradictionFinding]:
        """
        Heuristic-based contradiction detection (backward compatible).

        Args:
            statements: List of statements to check

        Returns:
            List of contradiction findings
        """
        findings: List[ContradictionFinding] = []

        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                first, second = statements[i], statements[j]
                if self._is_potential_conflict(first, second):
                    severity, rationale = self._evaluate_conflict(first, second)
                    findings.append(
                        ContradictionFinding(
                            statements=(first, second),
                            severity=severity,
                            description=rationale,
                            rationale=rationale
                        )
                    )

        return findings

    async def detect_with_llm_async(
        self,
        statements: List[Statement],
        skip_similar_check: bool = False
    ) -> List[ContradictionFinding]:
        """
        LLM-powered contradiction detection with semantic understanding.

        Pipeline:
        1. Fast filter - find candidate pairs using embeddings (if available)
        2. LLM verification - deep semantic analysis
        3. Resolution recommendations

        Args:
            statements: List of statements to check for contradictions
            skip_similar_check: Skip similarity filtering (check all pairs)

        Returns:
            List of contradiction findings with LLM explanations
        """
        if not self.use_llm:
            # Fall back to heuristic detection
            return self.detect(statements)

        if len(statements) < 2:
            return []

        # Step 1: Find candidate pairs (either all pairs or filtered by similarity)
        candidate_pairs: List[Tuple[Statement, Statement]] = []

        if skip_similar_check or not self.embedder:
            # Check all pairs
            for i in range(len(statements)):
                for j in range(i + 1, len(statements)):
                    if statements[i].source_id != statements[j].source_id:
                        candidate_pairs.append((statements[i], statements[j]))
        else:
            # Filter by similarity (statements about the same topic are more likely to contradict)
            candidate_pairs = await self._find_candidate_pairs(statements)

        # Step 2: LLM verification for each candidate
        contradictions: List[ContradictionFinding] = []

        for stmt1, stmt2 in candidate_pairs:
            try:
                result = await self._llm_check_contradiction(stmt1, stmt2)

                if result["is_contradiction"]:
                    contradictions.append(ContradictionFinding(
                        statements=(stmt1, stmt2),
                        severity=result.get("severity", "medium"),
                        description=result.get("explanation", "Contradiction detected"),
                        rationale=result.get("explanation", "Contradiction detected"),
                        resolution=result.get("resolution_strategy")
                    ))
            except Exception as e:
                print(f"LLM contradiction check failed: {e}")
                # Fall back to heuristic for this pair
                if self._is_potential_conflict(stmt1, stmt2):
                    severity, rationale = self._evaluate_conflict(stmt1, stmt2)
                    contradictions.append(ContradictionFinding(
                        statements=(stmt1, stmt2),
                        severity=severity,
                        description=rationale,
                        rationale=rationale
                    ))

        return contradictions

    async def _find_candidate_pairs(
        self,
        statements: List[Statement]
    ) -> List[Tuple[Statement, Statement]]:
        """Find candidate pairs using embedding similarity"""
        # This would use embeddings to find statements about similar topics
        # For now, fall back to checking all pairs
        candidates = []
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                if statements[i].source_id != statements[j].source_id:
                    # Simple keyword overlap as proxy for similarity
                    shared = self._shared_keywords(statements[i].text, statements[j].text)
                    if len(shared) >= 2:
                        candidates.append((statements[i], statements[j]))
        return candidates

    async def _llm_check_contradiction(
        self,
        stmt1: Statement,
        stmt2: Statement
    ) -> Dict[str, Any]:
        """
        Use LLM to determine if statements contradict.

        Returns:
            Dictionary with:
                - is_contradiction: bool
                - severity: "low" | "medium" | "high"
                - explanation: str
                - resolution_strategy: str
        """
        # Format authority for display
        auth1 = stmt1.authority if isinstance(stmt1.authority, str) else f"{stmt1.authority:.2f}"
        auth2 = stmt2.authority if isinstance(stmt2.authority, str) else f"{stmt2.authority:.2f}"

        prompt = f"""
Do these two statements contradict each other?

Statement 1: {stmt1.text}
Source: {stmt1.source_id} (Authority: {auth1})

Statement 2: {stmt2.text}
Source: {stmt2.source_id} (Authority: {auth2})

Analyze:
1. Direct contradiction? (one says X, other says NOT X)
2. Implication contradiction? (one implies X, other implies NOT X)
3. Temporal difference? (old information vs new information)
4. Different contexts/scopes? (both true but in different situations)

Respond with JSON only:
{{
    "is_contradiction": true/false,
    "severity": "low/medium/high",
    "explanation": "brief explanation of why they contradict or don't",
    "resolution_strategy": "which to trust and why, or how to reconcile"
}}
"""

        system_prompt = (
            "You are an expert at detecting contradictions and conflicts in information. "
            "Analyze carefully and return valid JSON only."
        )

        try:
            response = await self.llm_service.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Deterministic analysis
                max_tokens=300
            )

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                # Fallback if JSON parsing fails
                return {
                    "is_contradiction": False,
                    "severity": "low",
                    "explanation": "Could not parse LLM response",
                    "resolution_strategy": "Manual review recommended"
                }

        except Exception as e:
            print(f"LLM contradiction check error: {e}")
            return {
                "is_contradiction": False,
                "severity": "low",
                "explanation": f"Error: {str(e)}",
                "resolution_strategy": "Manual review recommended"
            }

    def _is_potential_conflict(self, first: Statement, second: Statement) -> bool:
        if first.source_id == second.source_id:
            return False
        # Basic polarity detection
        first_polarity = first.polarity if first.polarity is not None else self._infer_polarity(first.text)
        second_polarity = second.polarity if second.polarity is not None else self._infer_polarity(second.text)
        # Conflict if polarity differs and statements are semantically related
        if first_polarity == second_polarity:
            return False
        shared_tokens = self._shared_keywords(first.text, second.text)
        return len(shared_tokens) >= 2

    def _evaluate_conflict(self, first: Statement, second: Statement) -> Tuple[str, str]:
        first_authority = self.AUTHORITY_ORDER.get(first.authority.lower(), 0)
        second_authority = self.AUTHORITY_ORDER.get(second.authority.lower(), 0)
        authority_gap = abs(first_authority - second_authority)

        severity = "medium"
        if authority_gap >= 2:
            severity = "high"

        rationale = (
            f"Conflicting statements between {first.source_id} ({first.authority}) and "
            f"{second.source_id} ({second.authority}). Detected opposing claims."
        )

        if first.timestamp and second.timestamp:
            rationale += f" Timestamps: {first.timestamp} vs {second.timestamp}."

        return severity, rationale

    def _infer_polarity(self, text: str) -> int:
        lowered = text.lower()
        if any(marker in lowered for marker in self.NEGATION_MARKERS):
            return -1
        if "support" in lowered or "recommended" in lowered or "enabled" in lowered:
            return 1
        return 0

    @staticmethod
    def _shared_keywords(first: str, second: str) -> List[str]:
        first_tokens = {token.lower() for token in first.split() if len(token) > 3}
        second_tokens = {token.lower() for token in second.split() if len(token) > 3}
        return list(first_tokens & second_tokens)
