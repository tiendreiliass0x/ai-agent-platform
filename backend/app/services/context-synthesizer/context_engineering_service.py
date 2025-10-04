"""
Context Engineering Service - The Moat

Transforms scattered, messy data into well-structured knowledge packs
that power world-class expert agents.

This service is the competitive advantage:
- Intelligent data extraction and cleaning
- Semantic structuring and categorization
- Entity and relationship extraction
- Knowledge quality scoring
- Domain-specific optimization
- Continuous learning and refinement

"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.services.gemini_service import gemini_service
from app.services.embedding_service import embedding_service
from app.services.langextract_service import langextract_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge extracted from content"""
    PRODUCT_INFO = "product_info"           # Product features, specs, pricing
    PROCESS_GUIDE = "process_guide"         # How-to instructions, workflows
    POLICY = "policy"                       # Rules, policies, terms
    TROUBLESHOOTING = "troubleshooting"     # Problem-solution pairs
    FAQ = "faq"                            # Common questions and answers
    CONCEPT = "concept"                     # Definitions, explanations
    CASE_STUDY = "case_study"              # Examples, success stories
    COMPARISON = "comparison"               # Feature comparisons, alternatives
    INTEGRATION = "integration"             # API docs, technical integration
    BEST_PRACTICE = "best_practice"        # Recommendations, tips


class DataQuality(Enum):
    """Quality levels for extracted knowledge"""
    EXCELLENT = "excellent"    # 90-100% confidence, complete, well-structured
    GOOD = "good"             # 70-89% confidence, mostly complete
    FAIR = "fair"             # 50-69% confidence, some gaps
    POOR = "poor"             # <50% confidence, significant issues


@dataclass
class StructuredKnowledge:
    """Well-structured knowledge unit ready for agent consumption"""

    # Core Content
    title: str
    content: str
    knowledge_type: KnowledgeType

    # Semantic Structure
    key_entities: List[str] = field(default_factory=list)  # Products, features, people
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # Entity relationships
    categories: List[str] = field(default_factory=list)  # Hierarchical categories

    # Context
    prerequisites: List[str] = field(default_factory=list)  # What user needs to know first
    related_topics: List[str] = field(default_factory=list)  # Related knowledge units
    use_cases: List[str] = field(default_factory=list)  # When to use this knowledge

    # Quality Metrics
    quality_score: float = 0.0  # 0-1 quality rating
    quality_level: DataQuality = DataQuality.FAIR
    completeness: float = 0.0  # How complete is the information
    clarity: float = 0.0  # How clear and understandable

    # Metadata
    source_url: Optional[str] = None
    last_updated: Optional[datetime] = None
    language: str = "en"
    confidence: float = 0.0

    # Agent Optimization
    summary: str = ""  # Concise summary for quick reference
    key_points: List[str] = field(default_factory=list)  # Bullet points
    search_keywords: List[str] = field(default_factory=list)  # For retrieval

    # Embeddings
    embedding: Optional[List[float]] = None


@dataclass
class KnowledgePack:
    """Complete knowledge pack for an agent's domain"""

    pack_id: str
    name: str
    description: str
    domain: str  # e.g., "SaaS Sales", "Technical Support", "Product Education"

    # Structured Knowledge
    knowledge_units: List[StructuredKnowledge] = field(default_factory=list)

    # Organization
    category_tree: Dict[str, List[str]] = field(default_factory=dict)  # Hierarchical organization
    knowledge_graph: Dict[str, Dict] = field(default_factory=dict)  # Entity-relationship graph

    # Agent Guidance
    expert_tactics: Dict[str, Any] = field(default_factory=dict)  # How to use this knowledge
    response_templates: List[Dict] = field(default_factory=list)  # Proven response patterns

    # Quality Metrics
    overall_quality: float = 0.0
    coverage_score: float = 0.0  # How comprehensive
    freshness_score: float = 0.0  # How up-to-date

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"


class ContextEngineeringService:
    """
    The Moat: Transforms messy data into expert-grade knowledge packs
    """

    def __init__(self):
        self.gemini = gemini_service
        self.embeddings = embedding_service
        self.langextract = langextract_service

    async def engineer_knowledge_pack(
        self,
        documents: List[Dict[str, Any]],
        domain: str,
        agent_role: str = "expert",
        quality_threshold: float = 0.6
    ) -> KnowledgePack:
        """
        Main entry point: Transform raw documents into a structured knowledge pack

        Args:
            documents: List of raw documents with content and metadata
            domain: Domain area (e.g., "SaaS Sales", "Technical Support")
            agent_role: Role the agent will play
            quality_threshold: Minimum quality score to include knowledge

        Returns:
            Fully engineered KnowledgePack ready for agent use
        """
        logger.info(f"🏗️  Engineering knowledge pack for domain: {domain}")
        logger.info(f"   Processing {len(documents)} documents...")

        # Step 1: Extract and clean content
        extracted_units = await self._extract_knowledge_units(documents)
        logger.info(f"   ✓ Extracted {len(extracted_units)} knowledge units")

        # Step 2: Structure and categorize
        structured_units = await self._structure_knowledge(extracted_units, domain)
        logger.info(f"   ✓ Structured {len(structured_units)} units")

        # Step 3: Quality scoring and filtering
        quality_units = await self._score_and_filter(structured_units, quality_threshold)
        logger.info(f"   ✓ {len(quality_units)} units passed quality threshold ({quality_threshold})")

        # Step 4: Build knowledge graph and relationships
        knowledge_graph = await self._build_knowledge_graph(quality_units)
        logger.info(f"   ✓ Built knowledge graph with {len(knowledge_graph)} entities")

        # Step 5: Generate agent tactics and templates
        tactics = await self._generate_expert_tactics(quality_units, domain, agent_role)
        templates = await self._extract_response_patterns(quality_units)
        logger.info(f"   ✓ Generated {len(tactics)} tactics and {len(templates)} templates")

        # Step 6: Calculate pack-level metrics
        pack_metrics = self._calculate_pack_metrics(quality_units)

        # Build final knowledge pack
        pack = KnowledgePack(
            pack_id=f"pack_{domain.lower().replace(' ', '_')}_{int(datetime.utcnow().timestamp())}",
            name=f"{domain} Expert Knowledge",
            description=f"Engineered knowledge pack for {agent_role} in {domain}",
            domain=domain,
            knowledge_units=quality_units,
            knowledge_graph=knowledge_graph,
            expert_tactics=tactics,
            response_templates=templates,
            **pack_metrics
        )

        logger.info(f"✅ Knowledge pack engineered successfully!")
        logger.info(f"   Quality: {pack.overall_quality:.2%}")
        logger.info(f"   Coverage: {pack.coverage_score:.2%}")
        logger.info(f"   Units: {len(pack.knowledge_units)}")

        return pack

    async def _extract_knowledge_units(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract discrete knowledge units from raw documents"""

        all_units = []

        for doc in documents:
            content = doc.get("content", "")
            if not content or len(content.strip()) < 100:
                continue

            # Use LLM to intelligently chunk and extract knowledge
            prompt = f"""Analyze this document and extract discrete knowledge units.
Each unit should be self-contained and cover one specific topic, process, or concept.

Document:
{content[:4000]}  # Limit for prompt

Extract knowledge units in this JSON format:
[{{
  "title": "Clear, specific title",
  "content": "Complete, self-contained content",
  "type": "product_info|process_guide|policy|troubleshooting|faq|concept|case_study|comparison|integration|best_practice",
  "key_points": ["point 1", "point 2"],
  "entities": ["entity1", "entity2"],
  "confidence": 0.85
}}]

Focus on quality over quantity. Only extract complete, useful units."""

            try:
                response = await self.gemini.generate_content_async(prompt)
                units = self._parse_json_response(response)

                # Add source metadata
                for unit in units:
                    unit["source_url"] = doc.get("source_url")
                    unit["doc_id"] = doc.get("id")
                    unit["language"] = doc.get("language", "en")

                all_units.extend(units)

            except Exception as e:
                logger.warning(f"Failed to extract from document {doc.get('id')}: {e}")
                # Fallback: simple paragraph splitting
                fallback_units = self._fallback_extraction(content, doc)
                all_units.extend(fallback_units)

        return all_units

    async def _structure_knowledge(
        self,
        raw_units: List[Dict[str, Any]],
        domain: str
    ) -> List[StructuredKnowledge]:
        """Transform raw units into well-structured knowledge"""

        structured = []

        # Batch process for efficiency
        batch_size = 10
        for i in range(0, len(raw_units), batch_size):
            batch = raw_units[i:i+batch_size]

            tasks = [self._structure_single_unit(unit, domain) for unit in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, StructuredKnowledge):
                    structured.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Failed to structure unit: {result}")

        return structured

    async def _structure_single_unit(
        self,
        unit: Dict[str, Any],
        domain: str
    ) -> StructuredKnowledge:
        """Structure a single knowledge unit with rich metadata"""

        content = unit.get("content", "")
        title = unit.get("title", "")

        # Extract entities using language extraction service
        entities_data = await self.langextract.extract_entities(content)
        entities = [e["text"] for e in entities_data.get("entities", [])]

        # Determine knowledge type
        knowledge_type = self._classify_knowledge_type(unit.get("type"), content)

        # Generate semantic categories
        categories = await self._generate_categories(content, domain)

        # Extract relationships
        relationships = await self._extract_relationships(content, entities)

        # Generate embedding
        embedding = await self.embeddings.get_embedding(f"{title}\n\n{content}")

        # Quality scoring
        quality_metrics = await self._score_quality(content, title, entities)

        # Generate summary and key points
        summary = await self._generate_summary(content)
        key_points = await self._extract_key_points(content)

        # Generate search keywords
        keywords = await self._generate_keywords(title, content, entities)

        return StructuredKnowledge(
            title=title,
            content=content,
            knowledge_type=knowledge_type,
            key_entities=entities,
            relationships=relationships,
            categories=categories,
            quality_score=quality_metrics["overall"],
            quality_level=self._determine_quality_level(quality_metrics["overall"]),
            completeness=quality_metrics["completeness"],
            clarity=quality_metrics["clarity"],
            source_url=unit.get("source_url"),
            last_updated=datetime.utcnow(),
            language=unit.get("language", "en"),
            confidence=unit.get("confidence", 0.7),
            summary=summary,
            key_points=key_points,
            search_keywords=keywords,
            embedding=embedding
        )

    def _classify_knowledge_type(self, suggested_type: Optional[str], content: str) -> KnowledgeType:
        """Classify the type of knowledge"""

        # If type was suggested, try to use it
        if suggested_type:
            try:
                return KnowledgeType(suggested_type)
            except ValueError:
                pass

        # Classify based on content patterns
        content_lower = content.lower()

        if any(word in content_lower for word in ["how to", "step", "guide", "tutorial"]):
            return KnowledgeType.PROCESS_GUIDE
        elif any(word in content_lower for word in ["error", "troubleshoot", "fix", "problem", "issue"]):
            return KnowledgeType.TROUBLESHOOTING
        elif any(word in content_lower for word in ["pricing", "features", "product", "specification"]):
            return KnowledgeType.PRODUCT_INFO
        elif any(word in content_lower for word in ["policy", "terms", "agreement", "rule"]):
            return KnowledgeType.POLICY
        elif "?" in content and len(content) < 500:
            return KnowledgeType.FAQ
        elif any(word in content_lower for word in ["case study", "example", "success story"]):
            return KnowledgeType.CASE_STUDY
        elif any(word in content_lower for word in ["vs", "versus", "compared to", "alternative"]):
            return KnowledgeType.COMPARISON
        elif any(word in content_lower for word in ["api", "integration", "endpoint", "webhook"]):
            return KnowledgeType.INTEGRATION
        elif any(word in content_lower for word in ["best practice", "recommendation", "tip"]):
            return KnowledgeType.BEST_PRACTICE
        else:
            return KnowledgeType.CONCEPT

    async def _generate_categories(self, content: str, domain: str) -> List[str]:
        """Generate hierarchical categories for the knowledge"""

        prompt = f"""Given this content in the {domain} domain, suggest 2-4 hierarchical categories.
Categories should be specific and useful for organizing knowledge.

Content: {content[:500]}

Return ONLY a JSON array of category strings, like:
["Parent Category", "Sub Category", "Specific Topic"]"""

        try:
            response = await self.gemini.generate_content_async(prompt)
            categories = self._parse_json_response(response)
            return categories if isinstance(categories, list) else []
        except:
            # Fallback: extract from content
            return self._extract_categories_heuristic(content, domain)

    async def _extract_relationships(
        self,
        content: str,
        entities: List[str]
    ) -> Dict[str, List[str]]:
        """Extract entity relationships"""

        relationships = {}

        # Simple relationship extraction based on proximity and patterns
        for entity in entities[:20]:  # Limit for performance
            related = []

            # Find entities mentioned near this one
            for other_entity in entities:
                if other_entity != entity:
                    # Check if they appear close together in content
                    if self._entities_related(entity, other_entity, content):
                        related.append(other_entity)

            if related:
                relationships[entity] = related[:5]  # Top 5 related

        return relationships

    def _entities_related(self, entity1: str, entity2: str, content: str) -> bool:
        """Check if two entities are related based on proximity"""

        # Find positions of both entities
        positions1 = [m.start() for m in re.finditer(re.escape(entity1.lower()), content.lower())]
        positions2 = [m.start() for m in re.finditer(re.escape(entity2.lower()), content.lower())]

        # Check if any occurrences are within 200 characters of each other
        proximity_threshold = 200
        for pos1 in positions1:
            for pos2 in positions2:
                if abs(pos1 - pos2) < proximity_threshold:
                    return True

        return False

    async def _score_quality(
        self,
        content: str,
        title: str,
        entities: List[str]
    ) -> Dict[str, float]:
        """Score the quality of knowledge unit"""

        # Completeness score
        completeness = min(1.0, len(content) / 500)  # Ideal length ~500 chars
        if len(content) > 2000:
            completeness = 1.0

        # Clarity score (based on structure)
        has_title = bool(title and len(title) > 5)
        has_entities = len(entities) > 2
        has_structure = bool(re.search(r'\n\n|\n-|\n\d\.', content))

        clarity = (
            (0.4 if has_title else 0) +
            (0.3 if has_entities else 0) +
            (0.3 if has_structure else 0)
        )

        # Overall quality (weighted average)
        overall = (completeness * 0.5) + (clarity * 0.5)

        return {
            "completeness": completeness,
            "clarity": clarity,
            "overall": overall
        }

    def _determine_quality_level(self, score: float) -> DataQuality:
        """Determine quality level from score"""
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

    async def _generate_summary(self, content: str) -> str:
        """Generate concise summary"""

        if len(content) < 200:
            return content

        prompt = f"""Summarize this in 1-2 sentences:

{content[:1000]}

Summary:"""

        try:
            summary = await self.gemini.generate_content_async(prompt)
            return summary.strip()
        except:
            # Fallback: first sentence
            sentences = content.split('. ')
            return sentences[0] + '.' if sentences else content[:200]

    async def _extract_key_points(self, content: str) -> List[str]:
        """Extract key bullet points"""

        prompt = f"""Extract 3-5 key bullet points from this content:

{content[:1000]}

Return as JSON array of strings."""

        try:
            response = await self.gemini.generate_content_async(prompt)
            points = self._parse_json_response(response)
            return points if isinstance(points, list) else []
        except:
            # Fallback: extract existing bullets or create from sentences
            bullets = re.findall(r'[-•]\s*(.+)', content)
            return bullets[:5] if bullets else []

    async def _generate_keywords(
        self,
        title: str,
        content: str,
        entities: List[str]
    ) -> List[str]:
        """Generate search keywords"""

        keywords = set()

        # Add title words
        keywords.update(word.lower() for word in title.split() if len(word) > 3)

        # Add entities
        keywords.update(entity.lower() for entity in entities[:10])

        # Extract important terms from content
        words = re.findall(r'\b[A-Z][a-z]{3,}\b', content)  # Capitalized words
        keywords.update(word.lower() for word in words[:20])

        return list(keywords)[:30]  # Limit to 30 keywords

    async def _score_and_filter(
        self,
        units: List[StructuredKnowledge],
        threshold: float
    ) -> List[StructuredKnowledge]:
        """Filter knowledge units by quality threshold"""

        return [unit for unit in units if unit.quality_score >= threshold]

    async def _build_knowledge_graph(
        self,
        units: List[StructuredKnowledge]
    ) -> Dict[str, Dict]:
        """Build entity-relationship knowledge graph"""

        graph = {}

        # Aggregate all entities and their relationships
        for unit in units:
            for entity in unit.key_entities:
                if entity not in graph:
                    graph[entity] = {
                        "related_units": [],
                        "related_entities": set(),
                        "knowledge_types": set()
                    }

                graph[entity]["related_units"].append(unit.title)
                graph[entity]["knowledge_types"].add(unit.knowledge_type.value)

                # Add relationships
                if entity in unit.relationships:
                    graph[entity]["related_entities"].update(unit.relationships[entity])

        # Convert sets to lists for JSON serialization
        for entity in graph:
            graph[entity]["related_entities"] = list(graph[entity]["related_entities"])
            graph[entity]["knowledge_types"] = list(graph[entity]["knowledge_types"])

        return graph

    async def _generate_expert_tactics(
        self,
        units: List[StructuredKnowledge],
        domain: str,
        role: str
    ) -> Dict[str, Any]:
        """Generate expert tactics for using this knowledge"""

        # Analyze knowledge distribution
        type_distribution = {}
        for unit in units:
            knowledge_type = unit.knowledge_type.value
            type_distribution[knowledge_type] = type_distribution.get(knowledge_type, 0) + 1

        tactics = {
            "domain": domain,
            "role": role,
            "knowledge_distribution": type_distribution,
            "response_strategies": {
                "product_questions": "Leverage product_info and features knowledge",
                "troubleshooting": "Follow troubleshooting guides step-by-step",
                "how_to_questions": "Use process_guide knowledge with clear steps",
                "policy_questions": "Reference exact policy language",
                "comparisons": "Use comparison knowledge and feature matrices"
            },
            "confidence_thresholds": {
                "high_confidence": 0.8,  # Answer directly
                "medium_confidence": 0.6,  # Answer with caveats
                "low_confidence": 0.4  # Suggest escalation
            },
            "escalation_triggers": [
                "Knowledge not found",
                "Confidence < 0.4",
                "Conflicting information",
                "Request for pricing/legal advice"
            ]
        }

        return tactics

    async def _extract_response_patterns(
        self,
        units: List[StructuredKnowledge]
    ) -> List[Dict]:
        """Extract proven response patterns from knowledge"""

        templates = []

        # Group by knowledge type
        by_type = {}
        for unit in units:
            ktype = unit.knowledge_type.value
            if ktype not in by_type:
                by_type[ktype] = []
            by_type[ktype].append(unit)

        # Generate template for each type
        for ktype, type_units in by_type.items():
            if len(type_units) >= 3:  # Need multiple examples
                template = {
                    "knowledge_type": ktype,
                    "pattern": self._extract_pattern(type_units),
                    "example_count": len(type_units),
                    "avg_quality": sum(u.quality_score for u in type_units) / len(type_units)
                }
                templates.append(template)

        return templates

    def _extract_pattern(self, units: List[StructuredKnowledge]) -> str:
        """Extract common pattern from similar units"""

        # Simple pattern: structure of responses
        has_intro = sum(1 for u in units if "introduction" in u.content.lower()[:200])
        has_steps = sum(1 for u in units if re.search(r'\n\d\.|\n-', u.content))
        has_conclusion = sum(1 for u in units if "conclusion" in u.content.lower()[-200:])

        pattern_parts = []
        if has_intro > len(units) / 2:
            pattern_parts.append("introduction")
        if has_steps > len(units) / 2:
            pattern_parts.append("step-by-step content")
        if has_conclusion > len(units) / 2:
            pattern_parts.append("conclusion/summary")

        return " → ".join(pattern_parts) if pattern_parts else "freeform content"

    def _calculate_pack_metrics(self, units: List[StructuredKnowledge]) -> Dict[str, float]:
        """Calculate pack-level quality metrics"""

        if not units:
            return {
                "overall_quality": 0.0,
                "coverage_score": 0.0,
                "freshness_score": 0.0
            }

        # Overall quality: average of unit qualities
        overall_quality = sum(u.quality_score for u in units) / len(units)

        # Coverage: how many knowledge types are represented
        unique_types = len(set(u.knowledge_type for u in units))
        total_types = len(KnowledgeType)
        coverage_score = unique_types / total_types

        # Freshness: based on last_updated dates
        now = datetime.utcnow()
        avg_age_days = sum(
            (now - u.last_updated).days if u.last_updated else 365
            for u in units
        ) / len(units)
        freshness_score = max(0.0, 1.0 - (avg_age_days / 365))

        return {
            "overall_quality": overall_quality,
            "coverage_score": coverage_score,
            "freshness_score": freshness_score
        }

    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response"""
        import json

        # Try to extract JSON from response
        # Look for JSON array or object
        json_match = re.search(r'(\[.*\]|\{.*\})', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Fallback: return empty list
        return []

    def _fallback_extraction(
        self,
        content: str,
        doc: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback: Simple paragraph-based extraction"""

        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]

        units = []
        for i, para in enumerate(paragraphs[:20]):  # Max 20 units per doc
            # Try to extract a title from first line
            lines = para.split('\n')
            title = lines[0] if lines[0].isupper() or len(lines[0]) < 100 else f"Section {i+1}"

            units.append({
                "title": title,
                "content": para,
                "type": "concept",
                "key_points": [],
                "entities": [],
                "confidence": 0.5,
                "source_url": doc.get("source_url"),
                "doc_id": doc.get("id")
            })

        return units

    def _extract_categories_heuristic(self, content: str, domain: str) -> List[str]:
        """Heuristic category extraction"""

        categories = [domain]

        # Look for common category patterns
        content_lower = content.lower()

        if "setup" in content_lower or "getting started" in content_lower:
            categories.append("Getting Started")
        elif "advanced" in content_lower:
            categories.append("Advanced")
        elif "troubleshoot" in content_lower:
            categories.append("Troubleshooting")
        elif "api" in content_lower:
            categories.append("API & Integration")

        return categories


# Global instance
context_engineering_service = ContextEngineeringService()
