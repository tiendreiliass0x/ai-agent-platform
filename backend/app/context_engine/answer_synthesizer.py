"""
Answer Synthesis - Multi-source Fusion with LLM

Combines retrieved evidence into a coherent response:
    - LLM-powered multi-source synthesis
    - Consensus building and citation aggregation
    - Contradiction handling and transparency
    - Uncertainty notes for low-confidence answers
    - Reasoning transparency
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import textwrap

if TYPE_CHECKING:
    from ..services.llm_service_interface import LLMServiceInterface


@dataclass
class ContradictionFinding:
    """Represents a contradiction detected between sources"""
    statements: tuple[str, str]
    severity: str  # "low", "medium", "high"
    rationale: str
    resolution: Optional[str] = None


@dataclass
class SynthesizedAnswer:
    """Structured answer with support, citations, and confidence"""
    summary: str
    supporting_points: List[str]
    citations: List[str]
    confidence: Optional[float] = None
    uncertainty_notes: Optional[str] = None
    reasoning_chain: Optional[List[Dict[str, Any]]] = None


class AnswerSynthesizer:
    """Fuse evidence into a coherent, cited answer using LLM."""

    def __init__(
        self,
        llm_service: Optional["LLMServiceInterface"] = None,
        use_llm: bool = True,
        temperature: float = 0.3
    ) -> None:
        """
        Initialize the Answer Synthesizer.

        Args:
            llm_service: LLM service for synthesis (Gemini, OpenAI, etc.)
            use_llm: Whether to use LLM synthesis (True) or fallback to template (False)
            temperature: LLM temperature for synthesis (0.3 = balanced, deterministic)
        """
        self.llm_service = llm_service
        self.use_llm = use_llm and llm_service is not None
        self.temperature = temperature

    def synthesize(
        self,
        question: str,
        contexts: List[Dict[str, str]],
        confidence: Optional[float] = None,
    ) -> SynthesizedAnswer:
        """
        Synchronous synthesis (template-based fallback).

        Args:
            question: User's original question
            contexts: Retrieved context chunks with metadata
            confidence: Confidence score from ConfidenceScorer

        Returns:
            SynthesizedAnswer with summary, points, and citations
        """
        if not contexts:
            return SynthesizedAnswer(
                summary="I could not find relevant information. Could you provide more details?",
                supporting_points=[],
                citations=[],
                confidence=confidence,
            )

        supporting_points = self._build_supporting_points(contexts)
        summary = self._build_summary(question, supporting_points)
        citations = self._collect_citations(contexts)

        return SynthesizedAnswer(
            summary=summary,
            supporting_points=supporting_points,
            citations=citations,
            confidence=confidence,
        )

    async def synthesize_async(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        confidence: Optional[float] = None,
        reasoning_chain: Optional[List[Dict[str, Any]]] = None,
        contradictions: Optional[List[ContradictionFinding]] = None
    ) -> SynthesizedAnswer:
        """
        LLM-powered synthesis with contradiction handling and reasoning transparency.

        Args:
            question: User's original question
            contexts: Retrieved context chunks with metadata
            confidence: Confidence score from ConfidenceScorer
            reasoning_chain: Multi-hop reasoning steps (if any)
            contradictions: Detected contradictions (if any)

        Returns:
            SynthesizedAnswer with LLM-generated summary and structured support
        """
        if not contexts:
            return SynthesizedAnswer(
                summary="I could not find relevant information to answer your question. Could you provide more details or rephrase?",
                supporting_points=[],
                citations=[],
                confidence=confidence or 0.0,
                uncertainty_notes="No relevant context found in knowledge base"
            )

        # If LLM is available and enabled, use it
        if self.use_llm:
            try:
                return await self._synthesize_with_llm(
                    question,
                    contexts,
                    confidence,
                    reasoning_chain,
                    contradictions
                )
            except Exception as e:
                # Fall back to template-based synthesis
                print(f"LLM synthesis failed, using fallback: {e}")

        # Fallback: template-based synthesis
        return self.synthesize(question, contexts, confidence)

    async def _synthesize_with_llm(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        confidence: Optional[float],
        reasoning_chain: Optional[List[Dict[str, Any]]],
        contradictions: Optional[List[ContradictionFinding]]
    ) -> SynthesizedAnswer:
        """
        Use LLM to synthesize answer from multiple sources.

        This is the core Phase 2 enhancement - multi-source fusion with:
        - Consensus building when sources agree
        - Attribution when sources disagree
        - Contradiction handling
        - Uncertainty notes for low confidence
        """
        # Format contexts with source attribution
        context_text = self._format_contexts_for_llm(contexts)

        # Handle contradictions
        contradiction_notes = ""
        if contradictions and len(contradictions) > 0:
            contradiction_notes = self._format_contradictions(contradictions)

        # Reasoning transparency
        reasoning_notes = ""
        if reasoning_chain and len(reasoning_chain) > 0:
            reasoning_notes = self._format_reasoning(reasoning_chain)

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(
            question,
            context_text,
            confidence or 0.5,
            contradiction_notes,
            reasoning_notes
        )

        system_prompt = (
            "You are an expert information synthesizer. Your job is to combine evidence "
            "from multiple sources into a clear, accurate answer. Always cite sources. "
            "When sources disagree, present both views. Be transparent about uncertainty."
        )

        # Generate synthesis
        response = await self.llm_service.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=1000
        )

        # Parse LLM response
        return self._parse_llm_response(
            response,
            contexts,
            confidence,
            reasoning_chain
        )

    def _format_contexts_for_llm(self, contexts: List[Dict[str, Any]]) -> str:
        """Format contexts with source numbers for citation"""
        formatted = []
        for idx, ctx in enumerate(contexts, start=1):
            text = ctx.get("text", "")
            source = ctx.get("source_id") or ctx.get("chunk_id") or "unknown"
            authority = ctx.get("authority", "unverified")

            formatted.append(
                f"[Source {idx}] ({source}, authority: {authority})\n{text}\n"
            )

        return "\n".join(formatted)

    def _format_contradictions(self, contradictions: List[ContradictionFinding]) -> str:
        """Format contradictions for inclusion in prompt"""
        if not contradictions:
            return ""

        lines = ["⚠️ CONFLICTING INFORMATION DETECTED:"]
        for i, c in enumerate(contradictions, start=1):
            lines.append(
                f"{i}. Severity: {c.severity}\n"
                f"   Conflict: {c.rationale}\n"
                f"   Resolution: {c.resolution or 'Unresolved - present both views'}\n"
            )

        return "\n".join(lines)

    def _format_reasoning(self, reasoning_chain: List[Dict[str, Any]]) -> str:
        """Format reasoning chain for transparency"""
        if not reasoning_chain:
            return ""

        lines = ["REASONING STEPS TAKEN:"]
        for i, step in enumerate(reasoning_chain, start=1):
            gaps = step.get("gaps_identified", [])
            queries = step.get("sub_queries", [])

            lines.append(
                f"Step {i}: "
                f"Gaps identified: {', '.join(gaps) if gaps else 'none'}. "
                f"Retrieved: {len(step.get('new_evidence', []))} additional sources"
            )

        return "\n".join(lines)

    def _build_synthesis_prompt(
        self,
        question: str,
        context_text: str,
        confidence: float,
        contradiction_notes: str,
        reasoning_notes: str
    ) -> str:
        """Build the synthesis prompt for LLM"""

        uncertainty_instruction = ""
        if confidence < 0.7:
            uncertainty_instruction = (
                f"\n⚠️ CONFIDENCE IS LOW ({confidence:.0%}). "
                "Explicitly acknowledge uncertainty in your answer. "
                "Note what information might be missing or unclear."
            )

        prompt = f"""
Answer the following question using ONLY the provided context sources.

Question: {question}

Available Context:
{context_text}

{contradiction_notes}

{reasoning_notes}

{uncertainty_instruction}

Instructions:
1. Synthesize a clear, accurate answer from the sources
2. If sources AGREE, state the consensus confidently
3. If sources DISAGREE, present both views with clear attribution
4. Cite sources using [Source N] notation after each claim
5. Do NOT include information not present in the sources
6. If confidence is below 70%, add uncertainty notes

Required Format:
**Answer**: <your synthesized answer with inline citations>

**Key Points**:
- <Point 1 with [Source N] citation>
- <Point 2 with [Source N] citation>
- <Point 3 with [Source N] citation>

**Confidence Notes**: <only if confidence < 70% - what's uncertain or missing>

**Citations**: <list all sources used, e.g., [Source 1: source_id], [Source 2: source_id]>
"""
        return prompt.strip()

    def _parse_llm_response(
        self,
        response: str,
        contexts: List[Dict[str, Any]],
        confidence: Optional[float],
        reasoning_chain: Optional[List[Dict[str, Any]]]
    ) -> SynthesizedAnswer:
        """Parse the LLM's structured response"""

        # Extract sections using regex
        answer_match = re.search(r'\*\*Answer\*\*:?\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        points_match = re.search(r'\*\*Key Points\*\*:?\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        confidence_match = re.search(r'\*\*Confidence Notes\*\*:?\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        citations_match = re.search(r'\*\*Citations\*\*:?\s*(.+?)$', response, re.DOTALL)

        # Extract answer
        summary = answer_match.group(1).strip() if answer_match else response[:500]

        # Extract key points
        supporting_points = []
        if points_match:
            points_text = points_match.group(1).strip()
            # Split by bullet points or newlines
            points = re.findall(r'[-•*]\s*(.+)', points_text)
            supporting_points = [p.strip() for p in points if p.strip()]

        # Extract uncertainty notes
        uncertainty_notes = None
        if confidence_match:
            notes = confidence_match.group(1).strip()
            if notes and notes.lower() not in ["none", "n/a", "-"]:
                uncertainty_notes = notes

        # Extract citations
        citations = []
        if citations_match:
            citations_text = citations_match.group(1).strip()
            # Extract source references
            source_refs = re.findall(r'\[Source \d+:?\s*([^\]]+)\]', citations_text)
            citations = [ref.strip() for ref in source_refs]

        # Fallback: extract citations from contexts
        if not citations:
            citations = self._collect_citations(contexts)

        # Fallback: if no supporting points extracted, use template
        if not supporting_points:
            supporting_points = self._build_supporting_points(contexts)

        return SynthesizedAnswer(
            summary=summary,
            supporting_points=supporting_points,
            citations=citations,
            confidence=confidence,
            uncertainty_notes=uncertainty_notes,
            reasoning_chain=reasoning_chain
        )

    # Template-based helper methods (fallback)

    def _build_supporting_points(self, contexts: List[Dict[str, str]]) -> List[str]:
        """Template-based supporting point extraction (fallback)"""
        points = []
        for context in contexts:
            snippet = context.get("text", "")
            source = context.get("source_id") or context.get("chunk_id") or "unknown"
            sentence = snippet.strip().split("\n")[0]
            if sentence:
                points.append(f"{sentence.strip()} (Source: {source})")
        return points[:5]

    def _build_summary(self, question: str, points: List[str]) -> str:
        """Template-based summary generation (fallback)"""
        if not points:
            return "Unable to synthesize an answer from the available evidence."
        intro = f"In response to \"{question}\":"
        bullet_summary = " ".join(point.split(" (Source:")[0] for point in points[:3])
        summary = f"{intro} {bullet_summary}"
        return textwrap.shorten(summary, width=400, placeholder="...")

    def _collect_citations(self, contexts: List[Dict[str, str]]) -> List[str]:
        """Extract citations from contexts"""
        citations: List[str] = []
        for context in contexts:
            source_id = context.get("source_id") or context.get("source", "unknown")
            section = context.get("section")
            citation = source_id
            if section:
                citation = f"{source_id} ({section})"
            if citation not in citations:
                citations.append(citation)
        return citations[:5]
