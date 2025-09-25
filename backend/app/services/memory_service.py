"""
Advanced Memory Service for intelligent customer context management.
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
from ..models.customer_profile import CustomerProfile
from ..models.customer_memory import CustomerMemory, MemoryType, MemoryImportance
from ..models.conversation import Conversation
from ..models.message import Message
from ..services.database_service import db_service


class MemoryService:
    """Advanced memory management for intelligent agents"""

    def __init__(self):
        pass

    async def get_session(self) -> AsyncSession:
        """Get database session"""
        from ..core.database import get_db_session
        return await get_db_session()

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
        tags: List[str] = None
    ) -> CustomerMemory:
        """Store a new memory entry"""
        async with await self.get_session() as db:
            # Check if similar memory exists
            existing_memory = await self._find_similar_memory(
                customer_profile_id, key, memory_type
            )

            if existing_memory:
                # Update existing memory
                existing_memory.value = value
                existing_memory.confidence_score = max(existing_memory.confidence_score, confidence_score)
                existing_memory.last_accessed = datetime.utcnow()
                existing_memory.access_count += 1
                if context:
                    existing_memory.context.update(context)
                if tags:
                    for tag in tags:
                        existing_memory.add_tag(tag)

                await db.commit()
                await db.refresh(existing_memory)
                return existing_memory

            # Create new memory
            memory = CustomerMemory(
                customer_profile_id=customer_profile_id,
                conversation_id=conversation_id,
                memory_type=memory_type.value,
                key=key,
                value=value,
                importance=importance.value,
                confidence_score=confidence_score,
                context=context or {},
                tags=tags or []
            )

            db.add(memory)
            await db.commit()
            await db.refresh(memory)
            return memory

    async def retrieve_memories(
        self,
        customer_profile_id: int,
        memory_types: List[MemoryType] = None,
        tags: List[str] = None,
        min_confidence: float = 0.3,
        limit: int = 50
    ) -> List[CustomerMemory]:
        """Retrieve relevant memories for context"""
        async with await self.get_session() as db:
            query = select(CustomerMemory).where(
                and_(
                    CustomerMemory.customer_profile_id == customer_profile_id,
                    CustomerMemory.is_active == True,
                    CustomerMemory.confidence_score >= min_confidence
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

            # Update access tracking
            for memory in memories:
                memory.access()

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


# Global instance
memory_service = MemoryService()