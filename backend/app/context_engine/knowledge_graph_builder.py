"""
Knowledge Graph Builder - Entity Extraction & Relationships

Constructs a knowledge graph from documents by extracting:
- Entities (people, places, concepts, products)
- Relationships (causes, enables, contradicts, extends)
- Temporal links (supersedes, evolves_from)
- Conditional edges (applies_when, except_if)

Approach:
1. Pattern-based entity extraction (proper nouns, quoted terms)
2. Optional NER with spacy for advanced extraction
3. Co-occurrence and pattern-based relationship detection
4. Graph construction and querying for context retrieval
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re


@dataclass
class Entity:
    """An entity in the knowledge graph"""
    id: str
    type: str  # person, organization, product, concept, etc.
    name: str
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    mentions: int = 0  # How many times this entity appears

    def __hash__(self):
        return hash(self.id)


@dataclass
class Relationship:
    """A relationship between entities"""
    source_id: str
    target_id: str
    relation_type: str  # works_at, produces, uses, co_occurs, etc.
    confidence: float
    evidence: List[str] = field(default_factory=list)  # Text snippets supporting this
    frequency: int = 1  # How often this relationship appears


class KnowledgeGraphBuilder:
    """
    Build and query a knowledge graph from text.

    Approach:
    1. Extract entities using pattern matching and NER
    2. Detect relationships via co-occurrence and patterns
    3. Build graph structure
    4. Query subgraphs for context retrieval
    """

    def __init__(
        self,
        use_ner: bool = False,  # Use NLP NER model (requires spacy)
        co_occurrence_window: int = 100,  # Characters for co-occurrence
        min_confidence: float = 0.3
    ):
        """
        Initialize knowledge graph builder.

        Args:
            use_ner: Use advanced NER (requires spacy)
            co_occurrence_window: Window size for co-occurrence relationships
            min_confidence: Minimum confidence for relationships
        """
        self.use_ner = use_ner
        self._nlp_model = None  # Lazy load
        self.co_occurrence_window = co_occurrence_window
        self.min_confidence = min_confidence

        self.entities = {}  # entity_id -> Entity
        self.relationships = []  # List of Relationship objects
        self.entity_index = defaultdict(list)  # entity_name -> [entity_ids]

    @property
    def nlp_model(self):
        """Lazy load spacy NER model"""
        if self._nlp_model is None and self.use_ner:
            try:
                import spacy
                self._nlp_model = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                print("Warning: spacy not installed or model not found. Using pattern-based NER.")
                self.use_ner = False
        return self._nlp_model

    def extract_entities_pattern_based(self, text: str) -> List[Entity]:
        """
        Extract entities using simple pattern matching.

        Patterns:
        - Capitalized words (likely proper nouns)
        - Technical terms in quotes or backticks
        - Common entity markers (e.g., "CEO", "product", "API")

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        entities = []

        # Pattern 1: Capitalized phrases (proper nouns)
        # Match 2-4 capitalized words in a row
        capitalized_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b'
        matches = re.finditer(capitalized_pattern, text)

        for match in matches:
            name = match.group(1)
            # Skip common words
            if name.lower() in ['the', 'this', 'that', 'these', 'those', 'with', 'from']:
                continue

            entity_id = self._generate_entity_id(name)
            entities.append(Entity(
                id=entity_id,
                type="named_entity",
                name=name,
                attributes={"extraction_method": "pattern"}
            ))

        # Pattern 2: Quoted terms (products, features, concepts)
        quoted_pattern = r'["\']([^"\']+)["\']'
        matches = re.finditer(quoted_pattern, text)

        for match in matches:
            name = match.group(1)
            if len(name) < 3 or len(name) > 50:  # Skip too short/long
                continue

            entity_id = self._generate_entity_id(name)
            entities.append(Entity(
                id=entity_id,
                type="concept",
                name=name,
                attributes={"extraction_method": "quoted"}
            ))

        # Pattern 3: Technical terms in backticks
        backtick_pattern = r'`([^`]+)`'
        matches = re.finditer(backtick_pattern, text)

        for match in matches:
            name = match.group(1)
            entity_id = self._generate_entity_id(name)
            entities.append(Entity(
                id=entity_id,
                type="technical_term",
                name=name,
                attributes={"extraction_method": "backtick"}
            ))

        return entities

    def extract_entities_ner(self, text: str) -> List[Entity]:
        """
        Extract entities using spacy NER.

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        if self.nlp_model is None:
            return []

        doc = self.nlp_model(text)
        entities = []

        for ent in doc.ents:
            entity_type = ent.label_.lower()
            entity_id = self._generate_entity_id(ent.text)

            entities.append(Entity(
                id=entity_id,
                type=entity_type,
                name=ent.text,
                attributes={
                    "extraction_method": "ner",
                    "start": ent.start_char,
                    "end": ent.end_char
                }
            ))

        return entities

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text using available methods.

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        if self.use_ner and self.nlp_model is not None:
            return self.extract_entities_ner(text)
        else:
            return self.extract_entities_pattern_based(text)

    def extract_relationships_cooccurrence(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships based on co-occurrence.

        If two entities appear within a window, they likely have a relationship.

        Args:
            text: Input text
            entities: Known entities

        Returns:
            List of relationships
        """
        relationships = []

        # Build entity mention positions
        entity_positions = []
        for entity in entities:
            # Find all occurrences of this entity in text
            for match in re.finditer(re.escape(entity.name), text, re.IGNORECASE):
                entity_positions.append({
                    "entity": entity,
                    "start": match.start(),
                    "end": match.end()
                })

        # Sort by position
        entity_positions.sort(key=lambda x: x["start"])

        # Find co-occurring entities
        for i in range(len(entity_positions)):
            for j in range(i + 1, len(entity_positions)):
                e1 = entity_positions[i]
                e2 = entity_positions[j]

                # Check if within window
                distance = e2["start"] - e1["end"]
                if distance > self.co_occurrence_window:
                    break  # Sorted, so no need to check further

                # Extract evidence text
                evidence_start = max(0, e1["start"] - 20)
                evidence_end = min(len(text), e2["end"] + 20)
                evidence = text[evidence_start:evidence_end]

                # Calculate confidence based on distance
                # Closer entities = higher confidence
                confidence = max(0.3, min(1.0, 1.0 - (distance / self.co_occurrence_window)))

                relationships.append(Relationship(
                    source_id=e1["entity"].id,
                    target_id=e2["entity"].id,
                    relation_type="co_occurs",
                    confidence=confidence,
                    evidence=[evidence]
                ))

        return relationships

    def extract_relationships_pattern(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships using linguistic patterns.

        Patterns like "X works at Y", "X produces Y", etc.

        Args:
            text: Input text
            entities: Known entities

        Returns:
            List of relationships
        """
        relationships = []

        # Pattern: "X is a Y" (type-of relationship)
        # Pattern: "X works at Y" (employment)
        # Pattern: "X created Y" (creator)
        # Pattern: "X uses Y" (usage)

        relation_patterns = [
            (r'(.+?)\s+(?:is an?|are)\s+(.+?)(?:\.|,|\n)', 'is_a'),
            (r'(.+?)\s+(?:works? at|employed by)\s+(.+?)(?:\.|,|\n)', 'works_at'),
            (r'(.+?)\s+(?:created|developed|built)\s+(.+?)(?:\.|,|\n)', 'created'),
            (r'(.+?)\s+(?:uses?|utilizing)\s+(.+?)(?:\.|,|\n)', 'uses'),
            (r'(.+?)\s+(?:provides?|offers?)\s+(.+?)(?:\.|,|\n)', 'provides'),
        ]

        # Build entity name set for matching
        entity_names = {e.name.lower(): e for e in entities}

        for pattern, relation_type in relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()

                # Try to match with known entities
                source_entity = entity_names.get(source_text.lower())
                target_entity = entity_names.get(target_text.lower())

                if source_entity and target_entity:
                    relationships.append(Relationship(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        relation_type=relation_type,
                        confidence=0.8,
                        evidence=[match.group(0)]
                    ))

        return relationships

    def extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships between entities.

        Uses both co-occurrence and pattern-based methods.

        Args:
            text: Input text
            entities: Known entities in the text

        Returns:
            List of extracted relationships
        """
        # Co-occurrence relationships
        cooccur_rels = self.extract_relationships_cooccurrence(text, entities)

        # Pattern-based relationships
        pattern_rels = self.extract_relationships_pattern(text, entities)

        # Combine and deduplicate
        all_rels = cooccur_rels + pattern_rels

        return all_rels

    def _generate_entity_id(self, name: str) -> str:
        """Generate unique entity ID from name"""
        # Normalize name and create ID
        normalized = name.lower().strip().replace(' ', '_')
        normalized = re.sub(r'[^\w]', '', normalized)
        return f"entity_{normalized}"

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Merge duplicate entities (same name, different extractions).

        Args:
            entities: List of entities

        Returns:
            Deduplicated list
        """
        entity_map = {}

        for entity in entities:
            if entity.id in entity_map:
                # Merge
                existing = entity_map[entity.id]
                existing.mentions += 1
                # Merge aliases
                existing.aliases = list(set(existing.aliases + [entity.name]))
            else:
                entity_map[entity.id] = entity
                entity.aliases = list(set(entity.aliases + [entity.name]))
                entity.mentions = 1

        return list(entity_map.values())

    def build_graph(self, documents: List[str]) -> None:
        """
        Build knowledge graph from documents.

        Args:
            documents: List of text documents
        """
        all_entities = []
        all_relationships = []

        # Extract from each document
        for doc in documents:
            entities = self.extract_entities(doc)
            relationships = self.extract_relationships(doc, entities)

            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Merge duplicate entities
        merged_entities = self._merge_entities(all_entities)

        # Store in graph
        for entity in merged_entities:
            self.entities[entity.id] = entity
            self.entity_index[entity.name.lower()].append(entity.id)

        self.relationships = all_relationships

    def add_document(self, document_id: str, text: str) -> Dict[str, Any]:
        """
        Incrementally add a single document to the knowledge graph.

        Args:
            document_id: Identifier for the document (used in metadata only)
            text: Document text

        Returns:
            Entities and relationships extracted from this document
        """
        if not text:
            return {"entities": [], "relationships": []}

        raw_entities = self.extract_entities(text)
        deduped_entities = self._merge_entities(raw_entities)

        # Register entities and update indices
        new_entities = []
        for entity in deduped_entities:
            if entity.id in self.entities:
                existing = self.entities[entity.id]
                existing.mentions += entity.mentions
                existing.aliases = list(set(existing.aliases + entity.aliases))
            else:
                self.entities[entity.id] = entity
                new_entities.append(entity)

            self.entity_index[entity.name.lower()].append(entity.id)

        relationships = self.extract_relationships(text, deduped_entities)
        self.relationships.extend(relationships)

        return {
            "entities": deduped_entities,
            "relationships": relationships
        }

    def query_subgraph(
        self,
        entity_ids: List[str],
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Extract subgraph around entities.

        Args:
            entity_ids: Starting entity IDs
            max_hops: Maximum distance to traverse

        Returns:
            Subgraph with entities and relationships
        """
        visited_entities = set(entity_ids)
        subgraph_entities = {}
        subgraph_relationships = []

        # BFS traversal
        current_level = set(entity_ids)

        for hop in range(max_hops):
            next_level = set()

            # Find relationships involving current level
            for rel in self.relationships:
                if rel.source_id in current_level:
                    subgraph_relationships.append(rel)
                    next_level.add(rel.target_id)

                elif rel.target_id in current_level:
                    subgraph_relationships.append(rel)
                    next_level.add(rel.source_id)

            # Add newly discovered entities
            visited_entities.update(next_level)
            current_level = next_level

        # Collect all entities in subgraph
        for entity_id in visited_entities:
            if entity_id in self.entities:
                subgraph_entities[entity_id] = self.entities[entity_id]

        return {
            "entities": subgraph_entities,
            "relationships": subgraph_relationships
        }

    def find_entities_by_name(self, name: str) -> List[Entity]:
        """
        Find entities by name (case-insensitive).

        Args:
            name: Entity name

        Returns:
            List of matching entities
        """
        entity_ids = self.entity_index.get(name.lower(), [])
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_entity_connections(self, entity_id: str) -> List[Relationship]:
        """
        Get all relationships for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of relationships
        """
        connections = []

        for rel in self.relationships:
            if rel.source_id == entity_id or rel.target_id == entity_id:
                connections.append(rel)

        return connections

    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the knowledge graph.

        Returns:
            Summary dict with counts and statistics
        """
        entity_types = Counter(e.type for e in self.entities.values())
        relation_types = Counter(r.relation_type for r in self.relationships)

        return {
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "avg_connections_per_entity": len(self.relationships) / max(len(self.entities), 1)
        }
