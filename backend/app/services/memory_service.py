"""
Advanced Memory Service for intelligent customer context management.

Enhanced with governance and consent management for privacy compliance.
"""

import json
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import and_, or_, desc, func

from ..core.database import get_db
from ..core.governance import (
    ConsentContext, ConsentScope, DataRetentionPolicy, governance_engine,
    InferenceCategory, EvidenceItem
)
from ..models.customer_profile import CustomerProfile
from ..models.customer_memory import CustomerMemory, MemoryType, MemoryImportance
from ..models.conversation import Conversation
from ..models.message import Message
from ..services.database_service import db_service


class MemoryService:
    """Advanced memory management for intelligent agents with governance enforcement"""

    def __init__(self):
        self.memory_consent_mapping = {
            MemoryType.FACTUAL: ConsentScope.STORE_PREFERENCES,
            MemoryType.PREFERENCE: ConsentScope.STORE_PREFERENCES,
            MemoryType.BEHAVIORAL: ConsentScope.ANALYZE_BEHAVIOR,
            MemoryType.CONTEXTUAL: ConsentScope.USE_CONVERSATION_HISTORY,
            MemoryType.EPISODIC: ConsentScope.USE_CONVERSATION_HISTORY,
            MemoryType.PROCEDURAL: ConsentScope.ANALYZE_BEHAVIOR,
            MemoryType.EMOTIONAL: ConsentScope.ANALYZE_BEHAVIOR
        }

    async def get_session(self) -> AsyncSession:
        """Get database session"""
        from ..core.database import get_db_session
        return await get_db_session()

    def _validate_memory_consent(
        self,
        memory_type: MemoryType,
        consent_context: Optional[ConsentContext]
    ) -> bool:
        """Validate if we have consent to store this type of memory"""
        # If no consent context provided, create safe default for basic memory types
        if not consent_context:
            # Allow basic factual and preference storage with implicit consent
            if memory_type in [MemoryType.FACTUAL, MemoryType.PREFERENCE, MemoryType.CONTEXTUAL]:
                return True
            # Block sensitive memory types without explicit consent
            return False

        required_consent = self.memory_consent_mapping.get(memory_type)
        if not required_consent:
            return False

        return governance_engine.validate_consent(consent_context, required_consent)

    def _apply_retention_policy(
        self,
        memory_type: MemoryType,
        consent_context: Optional[ConsentContext]
    ) -> Tuple[DataRetentionPolicy, Optional[datetime]]:
        """Determine retention policy and expiration for memory"""
        if not consent_context:
            # Apply safe default retention based on memory type
            if memory_type in [MemoryType.FACTUAL, MemoryType.PREFERENCE]:
                return DataRetentionPolicy.SHORT_TERM, datetime.now() + timedelta(days=7)
            else:
                return DataRetentionPolicy.SESSION_ONLY, datetime.now() + timedelta(hours=1)

        base_policy = consent_context.data_retention_policy

        # Override for sensitive memory types
        if memory_type in [MemoryType.BEHAVIORAL, MemoryType.EMOTIONAL]:
            if base_policy == DataRetentionPolicy.LONG_TERM:
                base_policy = DataRetentionPolicy.MEDIUM_TERM

        # Calculate expiration based on policy
        now = datetime.now()
        if base_policy == DataRetentionPolicy.SESSION_ONLY:
            expiry = now + timedelta(hours=1)
        elif base_policy == DataRetentionPolicy.SHORT_TERM:
            expiry = now + timedelta(days=7)
        elif base_policy == DataRetentionPolicy.MEDIUM_TERM:
            expiry = now + timedelta(days=90)
        elif base_policy == DataRetentionPolicy.LONG_TERM:
            expiry = now + timedelta(days=365)
        else:
            expiry = now + timedelta(hours=1)  # Default to session only

        return base_policy, expiry

    def _redact_memory_content(
        self,
        content: str,
        consent_context: Optional[ConsentContext]
    ) -> str:
        """Apply PII redaction if required"""
        if not consent_context or not consent_context.pii_redaction_enabled:
            return content

        return governance_engine.redact_pii(content, hash_identifiers=True)

    # Customer Profile Management
    async def get_or_create_customer_profile(
        self,
        visitor_id: str,
        agent_id: int,
        initial_context: Dict[str, Any] = None
    ) -> CustomerProfile:
        """Get existing or create new customer profile"""
        async with await self.get_session() as db:
            # Try to find existing profile
            result = await db.execute(
                select(CustomerProfile).where(
                    and_(
                        CustomerProfile.visitor_id == visitor_id,
                        CustomerProfile.agent_id == agent_id
                    )
                )
            )
            profile = result.scalar_one_or_none()

            if profile:
                # Update last seen
                profile.last_seen_at = datetime.utcnow()
                await db.commit()
                return profile

            # Create new profile
            profile = CustomerProfile(
                visitor_id=visitor_id,
                agent_id=agent_id,
                device_info=initial_context.get('device_info', {}),
                location_data=initial_context.get('location_data', {}),
                last_page_visited=initial_context.get('page_url'),
                referrer_source=initial_context.get('referrer'),
                last_seen_at=datetime.utcnow()
            )

            db.add(profile)
            await db.commit()
            await db.refresh(profile)
            return profile

    async def update_customer_profile(
        self,
        customer_profile_id: int,
        updates: Dict[str, Any]
    ) -> Optional[CustomerProfile]:
        """Update customer profile with new information"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(CustomerProfile).where(CustomerProfile.id == customer_profile_id)
            )
            profile = result.scalar_one_or_none()

            if not profile:
                return None

            # Update fields
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)

            await db.commit()
            await db.refresh(profile)
            return profile

    # Memory Management
    async def store_memory(
        self,
        customer_profile_id: int,
        memory_type: MemoryType,
        key: str,
        value: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        context: Dict[str, Any] = None,
        conversation_id: int = None,
        confidence_score: float = 1.0,
        tags: List[str] = None,
        consent_context: Optional[ConsentContext] = None
    ) -> Optional[CustomerMemory]:
        """Store a new memory entry with governance enforcement"""

        # Governance validation
        if not self._validate_memory_consent(memory_type, consent_context):
            # Log blocked attempt for audit
            governance_engine.create_audit_log(
                action="memory_storage_blocked",
                customer_id=str(customer_profile_id),
                data_access={"memory_type": memory_type.value, "reason": "insufficient_consent"},
                consent_context=consent_context
            )
            return None

        # Apply retention policy
        retention_policy, expiry_date = self._apply_retention_policy(memory_type, consent_context)

        # Redact PII if required
        processed_value = self._redact_memory_content(value, consent_context)
        processed_key = self._redact_memory_content(key, consent_context)

        async with await self.get_session() as db:
            # Check if similar memory exists
            existing_memory = await self._find_similar_memory(
                customer_profile_id, processed_key, memory_type
            )

            if existing_memory:
                # Validate update consent
                if not governance_engine.should_persist_data(memory_type.value, consent_context):
                    return None

                # Update existing memory
                existing_memory.value = processed_value
                existing_memory.confidence_score = max(existing_memory.confidence_score, confidence_score)
                existing_memory.last_accessed = datetime.utcnow()
                existing_memory.access_count += 1
                existing_memory.valid_until = expiry_date

                if context:
                    existing_memory.context.update(context)
                if tags:
                    for tag in tags:
                        existing_memory.add_tag(tag)

                await db.commit()
                await db.refresh(existing_memory)

                # Create audit log
                governance_engine.create_audit_log(
                    action="memory_updated",
                    customer_id=str(customer_profile_id),
                    data_access={"memory_type": memory_type.value, "retention_policy": retention_policy.value},
                    consent_context=consent_context
                )

                return existing_memory

            # Create new memory with governance controls
            memory = CustomerMemory(
                customer_profile_id=customer_profile_id,
                conversation_id=conversation_id,
                memory_type=memory_type.value,
                key=processed_key,
                value=processed_value,
                importance=importance.value,
                confidence_score=confidence_score,
                context=context or {},
                tags=tags or [],
                valid_until=expiry_date,
                source=f"governed_storage_{retention_policy.value}"
            )

            db.add(memory)
            await db.commit()
            await db.refresh(memory)

            # Create audit log
            governance_engine.create_audit_log(
                action="memory_created",
                customer_id=str(customer_profile_id),
                data_access={"memory_type": memory_type.value, "retention_policy": retention_policy.value},
                consent_context=consent_context
            )

            return memory

    async def retrieve_memories(
        self,
        customer_profile_id: int,
        memory_types: List[MemoryType] = None,
        tags: List[str] = None,
        min_confidence: float = 0.3,
        limit: int = 50,
        consent_context: Optional[ConsentContext] = None
    ) -> List[CustomerMemory]:
        """Retrieve relevant memories for context with governance enforcement"""

        # Filter memory types based on consent
        if consent_context and memory_types:
            allowed_types = []
            for mt in memory_types:
                if self._validate_memory_consent(mt, consent_context):
                    allowed_types.append(mt)
            memory_types = allowed_types

        if memory_types is not None and not memory_types:  # No allowed types
            return []

        async with await self.get_session() as db:
            query = select(CustomerMemory).where(
                and_(
                    CustomerMemory.customer_profile_id == customer_profile_id,
                    CustomerMemory.is_active == True,
                    CustomerMemory.confidence_score >= min_confidence,
                    or_(
                        CustomerMemory.valid_until.is_(None),
                        CustomerMemory.valid_until > datetime.utcnow()
                    )
                )
            )

            if memory_types:
                type_values = [mt.value for mt in memory_types]
                query = query.where(CustomerMemory.memory_type.in_(type_values))

            if tags:
                # Filter by tags (JSON contains)
                for tag in tags:
                    query = query.where(CustomerMemory.tags.contains([tag]))

            query = query.order_by(desc(CustomerMemory.last_accessed)).limit(limit)

            result = await db.execute(query)
            memories = result.scalars().all()

            # Update access tracking and create audit log
            if memories:
                for memory in memories:
                    memory.access()

                # Log memory access for audit
                governance_engine.create_audit_log(
                    action="memory_accessed",
                    customer_id=str(customer_profile_id),
                    data_access={
                        "memory_count": len(memories),
                        "memory_types": list(set(m.memory_type for m in memories))
                    },
                    consent_context=consent_context
                )

            await db.commit()
            return memories

    async def get_contextual_memory(
        self,
        customer_profile_id: int,
        query_text: str,
        conversation_context: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive contextual memory for conversation"""

        # Get customer profile
        profile = await self._get_customer_profile(customer_profile_id)
        if not profile:
            return {}

        # Get relevant memories
        factual_memories = await self.retrieve_memories(
            customer_profile_id,
            memory_types=[MemoryType.FACTUAL, MemoryType.PREFERENCE],
            min_confidence=0.5,
            limit=10
        )

        behavioral_memories = await self.retrieve_memories(
            customer_profile_id,
            memory_types=[MemoryType.BEHAVIORAL, MemoryType.EMOTIONAL],
            min_confidence=0.4,
            limit=5
        )

        # Get recent conversation history
        recent_conversations = await self._get_recent_conversations(customer_profile_id, limit=3)

        # Compile context
        context = {
            "customer_profile": {
                "visitor_id": profile.visitor_id,
                "name": profile.display_name,
                "communication_style": profile.communication_style,
                "technical_level": profile.technical_level,
                "engagement_level": profile.engagement_level,
                "is_returning": profile.is_returning_customer,
                "primary_interests": profile.primary_interests or [],
                "pain_points": profile.pain_points or [],
                "journey_stage": profile.current_journey_stage
            },
            "factual_memories": [
                {
                    "key": m.key,
                    "value": m.value,
                    "confidence": m.confidence_score,
                    "importance": m.importance
                } for m in factual_memories
            ],
            "behavioral_insights": [
                {
                    "pattern": m.key,
                    "description": m.value,
                    "relevance": m.relevance_score
                } for m in behavioral_memories
            ],
            "conversation_history": recent_conversations,
            "personalization": {
                "preferred_style": profile.communication_style,
                "response_length": profile.response_length_preference,
                "technical_depth": profile.technical_level
            }
        }

        return context

    async def learn_from_conversation(
        self,
        customer_profile_id: int,
        conversation_id: int,
        messages: List[Dict[str, str]],
        session_metadata: Dict[str, Any] = None
    ) -> None:
        """Extract and store learnings from conversation"""

        # Analyze conversation for insights
        insights = await self._analyze_conversation_insights(messages)

        # Store factual information
        for fact in insights.get('facts', []):
            await self.store_memory(
                customer_profile_id=customer_profile_id,
                memory_type=MemoryType.FACTUAL,
                key=fact['key'],
                value=fact['value'],
                importance=MemoryImportance.MEDIUM,
                conversation_id=conversation_id,
                confidence_score=fact.get('confidence', 0.8)
            )

        # Store preferences
        for pref in insights.get('preferences', []):
            await self.store_memory(
                customer_profile_id=customer_profile_id,
                memory_type=MemoryType.PREFERENCE,
                key=pref['key'],
                value=pref['value'],
                importance=MemoryImportance.HIGH,
                conversation_id=conversation_id,
                confidence_score=pref.get('confidence', 0.7)
            )

        # Store behavioral patterns
        for behavior in insights.get('behaviors', []):
            await self.store_memory(
                customer_profile_id=customer_profile_id,
                memory_type=MemoryType.BEHAVIORAL,
                key=behavior['pattern'],
                value=behavior['description'],
                importance=MemoryImportance.MEDIUM,
                conversation_id=conversation_id,
                confidence_score=behavior.get('confidence', 0.6)
            )

        # Update customer profile
        profile_updates = {}
        if insights.get('communication_style'):
            profile_updates['communication_style'] = insights['communication_style']
        if insights.get('technical_level'):
            profile_updates['technical_level'] = insights['technical_level']
        if insights.get('interests'):
            # Add new interests
            profile = await self._get_customer_profile(customer_profile_id)
            if profile:
                for interest in insights['interests']:
                    profile.add_interest(interest)

        if profile_updates:
            await self.update_customer_profile(customer_profile_id, profile_updates)

    # Helper Methods
    async def _get_customer_profile(self, customer_profile_id: int) -> Optional[CustomerProfile]:
        """Get customer profile by ID"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(CustomerProfile).where(CustomerProfile.id == customer_profile_id)
            )
            return result.scalar_one_or_none()

    async def _find_similar_memory(
        self,
        customer_profile_id: int,
        key: str,
        memory_type: MemoryType
    ) -> Optional[CustomerMemory]:
        """Find similar existing memory"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(CustomerMemory).where(
                    and_(
                        CustomerMemory.customer_profile_id == customer_profile_id,
                        CustomerMemory.key == key,
                        CustomerMemory.memory_type == memory_type.value,
                        CustomerMemory.is_active == True
                    )
                )
            )
            return result.scalar_one_or_none()

    async def _get_recent_conversations(
        self,
        customer_profile_id: int,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Get recent conversation summaries"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Conversation)
                .options(selectinload(Conversation.messages))
                .where(Conversation.customer_profile_id == customer_profile_id)
                .order_by(desc(Conversation.created_at))
                .limit(limit)
            )
            conversations = result.scalars().all()

            return [
                {
                    "id": conv.id,
                    "date": conv.created_at.isoformat(),
                    "message_count": len(conv.messages),
                    "summary": self._summarize_conversation(conv.messages[:5])  # First 5 messages
                }
                for conv in conversations
            ]

    def _summarize_conversation(self, messages: List[Message]) -> str:
        """Create a brief conversation summary"""
        if not messages:
            return "No messages"

        user_messages = [m for m in messages if m.role == "user"]
        if user_messages:
            return f"Customer discussed: {user_messages[0].content[:100]}..."
        return "System interaction"

    async def _analyze_conversation_insights(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Analyze conversation for behavioral insights and facts"""

        # Simple pattern-based analysis (could be enhanced with LLM)
        insights = {
            'facts': [],
            'preferences': [],
            'behaviors': [],
            'interests': [],
            'communication_style': None,
            'technical_level': None
        }

        user_messages = [m['content'].lower() for m in messages if m['role'] == 'user']
        full_text = ' '.join(user_messages)

        # Detect communication style
        if any(word in full_text for word in ['please', 'thank you', 'appreciate', 'sorry']):
            insights['communication_style'] = 'formal'
        elif any(word in full_text for word in ['hey', 'hi', 'cool', 'awesome', 'yeah']):
            insights['communication_style'] = 'casual'

        # Detect technical level
        technical_terms = ['api', 'integration', 'endpoint', 'configuration', 'deployment']
        if any(term in full_text for term in technical_terms):
            insights['technical_level'] = 'expert'
        elif any(word in full_text for word in ['how to', 'what is', 'explain', 'simple']):
            insights['technical_level'] = 'beginner'

        # Detect interests
        interest_keywords = {
            'pricing': ['price', 'cost', 'billing', 'payment', 'subscription'],
            'features': ['feature', 'functionality', 'capability', 'what can'],
            'support': ['help', 'support', 'problem', 'issue', 'error'],
            'integration': ['integrate', 'connect', 'api', 'webhook']
        }

        for interest, keywords in interest_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                insights['interests'].append(interest)

        return insights

    # Governance-specific methods
    async def cleanup_expired_memories(self) -> Dict[str, int]:
        """Clean up expired memories based on retention policies"""
        async with await self.get_session() as db:
            # Find expired memories
            expired_query = select(CustomerMemory).where(
                and_(
                    CustomerMemory.valid_until.isnot(None),
                    CustomerMemory.valid_until <= datetime.utcnow(),
                    CustomerMemory.is_active == True
                )
            )

            result = await db.execute(expired_query)
            expired_memories = result.scalars().all()

            cleanup_count = len(expired_memories)
            retention_counts = {}

            for memory in expired_memories:
                # Mark as inactive instead of deleting for audit trail
                memory.is_active = False
                memory.invalidate()

                # Track by source for reporting
                source = memory.source or "unknown"
                retention_counts[source] = retention_counts.get(source, 0) + 1

            await db.commit()

            # Log cleanup for audit
            if cleanup_count > 0:
                governance_engine.create_audit_log(
                    action="memory_cleanup_expired",
                    customer_id="system",
                    data_access={
                        "cleaned_count": cleanup_count,
                        "retention_breakdown": retention_counts
                    }
                )

            return {
                "total_cleaned": cleanup_count,
                "by_retention_policy": retention_counts
            }

    async def revoke_consent_memories(
        self,
        customer_profile_id: int,
        revoked_consents: List[ConsentScope]
    ) -> int:
        """Remove memories that are no longer consented to"""
        async with await self.get_session() as db:
            # Find memories that require revoked consents
            affected_types = []
            for memory_type, required_consent in self.memory_consent_mapping.items():
                if required_consent in revoked_consents:
                    affected_types.append(memory_type.value)

            if not affected_types:
                return 0

            # Find affected memories
            affected_query = select(CustomerMemory).where(
                and_(
                    CustomerMemory.customer_profile_id == customer_profile_id,
                    CustomerMemory.memory_type.in_(affected_types),
                    CustomerMemory.is_active == True
                )
            )

            result = await db.execute(affected_query)
            affected_memories = result.scalars().all()

            # Mark as inactive
            for memory in affected_memories:
                memory.is_active = False
                memory.source = f"consent_revoked_{memory.source}"

            await db.commit()

            # Log consent revocation
            governance_engine.create_audit_log(
                action="memory_consent_revoked",
                customer_id=str(customer_profile_id),
                data_access={
                    "revoked_consents": [c.value for c in revoked_consents],
                    "affected_memories": len(affected_memories),
                    "memory_types": affected_types
                }
            )

            return len(affected_memories)

    async def get_memory_audit_trail(
        self,
        customer_profile_id: int,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get audit trail for customer memory operations"""
        async with await self.get_session() as db:
            # Get recent memory operations
            since_date = datetime.utcnow() - timedelta(days=days_back)

            # Get memory creation/updates
            query = select(CustomerMemory).where(
                and_(
                    CustomerMemory.customer_profile_id == customer_profile_id,
                    CustomerMemory.created_at >= since_date
                )
            ).order_by(desc(CustomerMemory.created_at))

            result = await db.execute(query)
            memories = result.scalars().all()

            audit_trail = []
            for memory in memories:
                audit_trail.append({
                    "timestamp": memory.created_at.isoformat(),
                    "action": "memory_created" if memory.is_active else "memory_deactivated",
                    "memory_type": memory.memory_type,
                    "importance": memory.importance,
                    "source": memory.source,
                    "retention_policy": "inferred_from_source"
                })

            return audit_trail


# Global instance
memory_service = MemoryService()