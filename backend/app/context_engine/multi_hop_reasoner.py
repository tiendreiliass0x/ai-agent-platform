"""
Multi-Hop Reasoning - Iterative Retrieval & Graph Traversal

Provides lightweight planning to answer complex questions by:
    - Generating reasoning steps from query decomposition
    - Traversing the knowledge graph to stitch related facts
    - Coordinating iterative retrieval (retrieve → reason → retrieve again)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .knowledge_graph_builder import KnowledgeGraphBuilder, Relationship


@dataclass
class ReasoningStep:
    description: str
    supporting_chunks: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)


@dataclass
class ReasoningPlan:
    steps: List[ReasoningStep]
    requires_additional_retrieval: bool = False


class MultiHopReasoner:
    """Coordinate multi-hop reasoning over retrieved evidence and the knowledge graph."""

    def __init__(self, knowledge_graph: Optional[KnowledgeGraphBuilder] = None) -> None:
        self.knowledge_graph = knowledge_graph or KnowledgeGraphBuilder()

    def plan(
        self,
        query: str,
        initial_chunks: Sequence[Dict[str, any]],
        entities: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> ReasoningPlan:
        """
        Build a reasoning plan combining retrieved chunks, entity relationships, and graph traversal.
        """
        entities = entities or self._collect_entities(initial_chunks)
        steps: List[ReasoningStep] = []

        if not initial_chunks:
            return ReasoningPlan(
                steps=[ReasoningStep(description="Perform initial retrieval for relevant evidence.")],
                requires_additional_retrieval=True,
            )

        # Step 1: Summarize evidence from initial chunks.
        supporting_ids = [chunk.get("chunk_id", f"chunk_{idx}") for idx, chunk in enumerate(initial_chunks)]
        steps.append(
            ReasoningStep(
                description="Summarize direct evidence from top-ranked chunks.",
                supporting_chunks=supporting_ids[:5],
                related_entities=entities[:5],
            )
        )

        # Step 2: Traverse knowledge graph for additional relationships.
        if entities:
            graph_context = self._traverse_graph(entities, max_hops=max_hops)
            if graph_context:
                for entity_id, rels in graph_context.items():
                    step_entities = [entity_id] + [rel.target_id for rel in rels]
                    steps.append(
                        ReasoningStep(
                            description=f"Inspect relationships around entity '{entity_id}'.",
                            supporting_chunks=[],
                            related_entities=step_entities,
                        )
                    )

        # Step 3: Determine if additional retrieval is needed.
        requires_more = len(steps) < 2 or any("verify" in chunk.get("text", "").lower() for chunk in initial_chunks)

        if requires_more:
            steps.append(
                ReasoningStep(
                    description="Perform follow-up retrieval for missing sub-questions or validations.",
                    supporting_chunks=[],
                    related_entities=[],
                )
            )

        return ReasoningPlan(steps=steps, requires_additional_retrieval=requires_more)

    def _collect_entities(self, chunks: Sequence[Dict[str, any]]) -> List[str]:
        entities: List[str] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            chunk_entities = metadata.get("entities") or []
            for entity in chunk_entities:
                if entity not in entities and len(entity) > 2:
                    entities.append(entity)
        return entities

    def _traverse_graph(self, entities: List[str], max_hops: int) -> Dict[str, List[Relationship]]:
        graph_context: Dict[str, List[Relationship]] = {}
        for entity in entities:
            matches = self.knowledge_graph.find_entities_by_name(entity)
            for match in matches:
                rels = self.knowledge_graph.get_entity_connections(match.id)
                if rels:
                    graph_context[match.id] = rels[: max_hops * 2]
        return graph_context
