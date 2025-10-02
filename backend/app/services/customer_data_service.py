"""
Customer Data Service - Handles customer data access, profile creation, and fallback strategies
for both known customers and generic users with no profile.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.customer_profile import CustomerProfile
from ..models.customer_memory import CustomerMemory, MemoryType
from ..models.conversation import Conversation
from ..models.agent import Agent
from .memory_service import memory_service
from .langextract_service import lang_extract_service


@dataclass
class CustomerContext:
    """Unified customer context for both known and unknown users"""
    # Identity
    customer_profile_id: Optional[int] = None
    visitor_id: str = None
    email: Optional[str] = None
    name: Optional[str] = None

    # Profile Data
    profile_type: str = "unknown"  # "unknown", "anonymous", "identified", "registered"
    engagement_level: str = "new"
    communication_style: str = "neutral"
    technical_level: str = "intermediate"

    # Session Data
    session_context: Dict[str, Any] = None
    current_interests: List[str] = None
    pain_points: List[str] = None
    goals: List[str] = None

    # Behavioral Insights
    conversation_history: List[Dict] = None
    preferences: Dict[str, Any] = None
    satisfaction_score: Optional[float] = None

    # Business Context
    journey_stage: str = "awareness"
    is_returning: bool = False
    last_interaction: Optional[datetime] = None

    # Context Quality
    confidence_score: float = 0.0  # How confident we are in our understanding
    data_sources: List[str] = None  # Where our data comes from
    sentiment: Optional[Dict[str, Any]] = None
    named_entities: List[Dict[str, Any]] = None


@dataclass
class ProductContext:
    """Product and business context"""
    # Agent/Organization Products
    product_catalog: List[Dict[str, Any]] = None
    featured_products: List[Dict[str, Any]] = None
    pricing_info: Dict[str, Any] = None
    promotions: List[Dict[str, Any]] = None

    # Business Knowledge
    company_info: Dict[str, Any] = None
    policies: Dict[str, Any] = None
    support_resources: List[Dict[str, Any]] = None
    faqs: List[Dict[str, Any]] = None


class CustomerDataService:
    """Handles all customer data access and context building"""

    def __init__(self):
        self.fallback_strategies = {
            "unknown": self._handle_unknown_user,
            "minimal": self._handle_minimal_context,
            "anonymous": self._handle_anonymous_user,
            "identified": self._handle_identified_user
        }

    async def get_customer_context(
        self,
        visitor_id: str,
        agent_id: int,
        session_context: Dict[str, Any],
        db: AsyncSession
    ) -> CustomerContext:
        """Get comprehensive customer context with fallback strategies"""

        # Try to find existing customer profile
        customer_profile = await self._get_existing_customer_profile(visitor_id, agent_id, db)

        if customer_profile:
            # Known customer - build rich context
            return await self._build_rich_customer_context(customer_profile, session_context, db)
        else:
            # Unknown user - use smart fallback strategies
            return await self._build_fallback_customer_context(visitor_id, agent_id, session_context, db)

    async def get_product_context(self, agent_id: int, db: AsyncSession) -> ProductContext:
        """Get product and business context for the agent's organization"""

        # Get agent and organization info
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

        if not agent:
            return ProductContext()

        # Build product context based on agent's knowledge base and organization
        return await self._build_product_context(agent, db)

    async def create_or_update_customer_profile(
        self,
        visitor_id: str,
        agent_id: int,
        session_context: Dict[str, Any],
        interaction_data: Dict[str, Any],
        db: AsyncSession
    ) -> CustomerProfile:
        """Create new customer profile or update existing one"""

        # Try to find existing profile
        existing_profile = await self._get_existing_customer_profile(visitor_id, agent_id, db)

        if existing_profile:
            # Update existing profile
            return await self._update_customer_profile(existing_profile, interaction_data, db)
        else:
            # Create new profile
            return await self._create_new_customer_profile(visitor_id, agent_id, session_context, interaction_data, db)

    async def enrich_with_langextract(
        self,
        message: str,
        customer_context: CustomerContext
    ) -> Optional[Dict[str, Any]]:
        analysis = await lang_extract_service.analyze_text(message)
        if analysis:
            customer_context.sentiment = analysis.sentiment
            customer_context.named_entities = analysis.entities
            return {
                "sentiment": analysis.sentiment,
                "entities": analysis.entities,
            }
        return None

    async def _get_existing_customer_profile(
        self,
        visitor_id: str,
        agent_id: int,
        db: AsyncSession
    ) -> Optional[CustomerProfile]:
        """Try to find existing customer profile"""

        # First try by visitor_id and agent_id
        result = await db.execute(
            select(CustomerProfile).where(
                CustomerProfile.visitor_id == visitor_id,
                CustomerProfile.agent_id == agent_id
            )
        )
        profile = result.scalar_one_or_none()

        if profile:
            return profile

        # If visitor_id looks like an email, try to find by email
        if "@" in visitor_id:
            result = await db.execute(
                select(CustomerProfile).where(
                    CustomerProfile.email == visitor_id,
                    CustomerProfile.agent_id == agent_id
                )
            )
            profile = result.scalar_one_or_none()

        return profile

    async def _build_rich_customer_context(
        self,
        customer_profile: CustomerProfile,
        session_context: Dict[str, Any],
        db: AsyncSession
    ) -> CustomerContext:
        """Build rich context for known customers"""

        # Get conversation history
        conversation_history = await self._get_conversation_history(customer_profile.id, db)

        # Get memory entries
        memory_entries = await memory_service.get_customer_memories(customer_profile.id)

        return CustomerContext(
            customer_profile_id=customer_profile.id,
            visitor_id=customer_profile.visitor_id,
            email=customer_profile.email,
            name=customer_profile.name,
            profile_type="identified" if customer_profile.email else "anonymous",
            engagement_level=customer_profile.engagement_level,
            communication_style=customer_profile.communication_style,
            technical_level=customer_profile.technical_level,
            session_context=session_context,
            current_interests=customer_profile.primary_interests or [],
            pain_points=customer_profile.pain_points or [],
            goals=customer_profile.goals or [],
            conversation_history=conversation_history,
            preferences=customer_profile.preferences or {},
            satisfaction_score=customer_profile.satisfaction_score,
            journey_stage=customer_profile.current_journey_stage,
            is_returning=customer_profile.is_returning_customer,
            last_interaction=customer_profile.last_seen_at,
            confidence_score=0.8,  # High confidence for known customers
            data_sources=["customer_profile", "conversation_history", "memory_entries"]
        )

    async def _build_fallback_customer_context(
        self,
        visitor_id: str,
        agent_id: int,
        session_context: Dict[str, Any],
        db: AsyncSession
    ) -> CustomerContext:
        """Build context for unknown users using smart fallbacks"""

        # Analyze session context for immediate insights
        inferred_data = await self._infer_customer_data_from_session(session_context)

        # Check if visitor ID suggests identity
        profile_type = self._classify_visitor_type(visitor_id)

        return CustomerContext(
            customer_profile_id=None,
            visitor_id=visitor_id,
            email=inferred_data.get("email"),
            name=inferred_data.get("name"),
            profile_type=profile_type,
            engagement_level="new",
            communication_style=inferred_data.get("communication_style", "neutral"),
            technical_level=inferred_data.get("technical_level", "intermediate"),
            session_context=session_context,
            current_interests=inferred_data.get("interests", []),
            pain_points=[],
            goals=[],
            conversation_history=[],
            preferences=inferred_data.get("preferences", {}),
            satisfaction_score=None,
            journey_stage="awareness",
            is_returning=False,
            last_interaction=None,
            confidence_score=0.3,  # Lower confidence for unknown users
            data_sources=["session_context", "inference"]
        )

    async def _build_product_context(self, agent: Agent, db: AsyncSession) -> ProductContext:
        """Build comprehensive product context"""

        # Get organization-specific product data
        # This would integrate with your knowledge base, CMS, or product catalog

        # For Coconut Furniture example:
        if agent.organization_id == 4:  # Coconut Furniture
            return ProductContext(
                product_catalog=[
                    {
                        "id": "sofa-modern-sectional",
                        "name": "Modern Sectional Sofa",
                        "category": "Living Room",
                        "price_range": "$1,200 - $2,800",
                        "features": ["Sustainable materials", "Modular design", "Premium comfort"],
                        "popular": True
                    },
                    {
                        "id": "dining-table-sustainable",
                        "name": "Sustainable Dining Table",
                        "category": "Dining Room",
                        "price_range": "$800 - $1,500",
                        "features": ["Reclaimed wood", "Modern design", "Seats 6-8"],
                        "popular": True
                    }
                ],
                featured_products=[
                    {
                        "name": "Eco-Friendly Collection",
                        "description": "Sustainable furniture made from reclaimed materials"
                    }
                ],
                pricing_info={
                    "financing_available": True,
                    "return_policy": "30 days",
                    "shipping": "Free on orders over $500"
                },
                promotions=[
                    {
                        "title": "Winter Sale",
                        "discount": "20% off living room furniture",
                        "expires": "2024-02-29"
                    }
                ],
                company_info={
                    "name": "Coconut Furniture",
                    "mission": "Premium sustainable furniture for modern homes",
                    "established": "2018",
                    "specialties": ["Sustainable materials", "Modern design", "Custom solutions"]
                },
                policies={
                    "return_policy": "30-day returns on all items",
                    "warranty": "2-year warranty on all furniture",
                    "shipping": "Free shipping on orders over $500"
                },
                faqs=[
                    {
                        "question": "Do you offer delivery and setup?",
                        "answer": "Yes, we offer white-glove delivery and setup for an additional fee."
                    },
                    {
                        "question": "Are your materials really sustainable?",
                        "answer": "All our wood is FSC-certified and we use low-VOC finishes."
                    }
                ]
            )

        # Default/generic product context
        return ProductContext(
            company_info={"name": "Our Company", "mission": "Helping customers succeed"},
            policies={"return_policy": "Standard return policy applies"},
            faqs=[
                {"question": "How can I contact support?", "answer": "You can reach us anytime through this chat."}
            ]
        )

    async def _infer_customer_data_from_session(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Infer customer characteristics from session data"""

        inferred = {}

        # Analyze URL patterns
        current_page = session_context.get("current_page", "")
        if "/pricing" in current_page:
            inferred["interests"] = ["pricing"]
            inferred["journey_stage"] = "consideration"
        elif "/support" in current_page:
            inferred["interests"] = ["support"]
            inferred["communication_style"] = "problem_solving"
        elif "/products" in current_page:
            inferred["interests"] = ["products"]
            inferred["journey_stage"] = "awareness"

        # Analyze device/browser
        user_agent = session_context.get("user_agent", "")
        if "mobile" in user_agent.lower():
            inferred["preferences"] = {"device": "mobile", "response_length": "brief"}

        # Analyze referrer
        referrer = session_context.get("referrer", "")
        if "google" in referrer:
            inferred["acquisition_channel"] = "search"
        elif "facebook" in referrer or "twitter" in referrer:
            inferred["acquisition_channel"] = "social"

        return inferred

    def _classify_visitor_type(self, visitor_id: str) -> str:
        """Classify visitor type based on visitor_id format"""

        if "@" in visitor_id:
            return "identified"  # Email address
        elif visitor_id.startswith("anon_"):
            return "anonymous"  # Anonymous session
        elif visitor_id.startswith("guest_"):
            return "guest"  # Guest user
        elif len(visitor_id) > 20:
            return "anonymous"  # Long random ID
        else:
            return "unknown"  # Unknown format

    async def _handle_unknown_user(self, context: CustomerContext) -> CustomerContext:
        """Handle completely unknown users"""

        # Set helpful defaults
        context.communication_style = "friendly"
        context.technical_level = "beginner"
        context.current_interests = ["general_information"]
        context.goals = ["learn_about_products"]

        return context

    async def _handle_minimal_context(self, context: CustomerContext) -> CustomerContext:
        """Handle users with minimal context"""

        # Use conservative assumptions
        context.communication_style = "professional"
        context.response_length_preference = "medium"

        return context

    async def _handle_anonymous_user(self, context: CustomerContext) -> CustomerContext:
        """Handle anonymous users with some session data"""

        # Try to build context from session
        if context.session_context:
            # Add session-based insights
            pass

        return context

    async def _handle_identified_user(self, context: CustomerContext) -> CustomerContext:
        """Handle identified users (email known)"""

        # Higher confidence in our understanding
        context.confidence_score = 0.6
        context.communication_style = "personalized"

        return context

    async def _get_conversation_history(self, customer_profile_id: int, db: AsyncSession, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for the customer"""

        # This would query your conversation/message tables
        # For now, return empty list
        return []

    async def _create_new_customer_profile(
        self,
        visitor_id: str,
        agent_id: int,
        session_context: Dict[str, Any],
        interaction_data: Dict[str, Any],
        db: AsyncSession
    ) -> CustomerProfile:
        """Create a new customer profile"""

        profile = CustomerProfile(
            visitor_id=visitor_id,
            agent_id=agent_id,
            email=interaction_data.get("email"),
            name=interaction_data.get("name"),
            communication_style=interaction_data.get("communication_style", "neutral"),
            primary_interests=interaction_data.get("interests", []),
            current_journey_stage="awareness",
            device_info=session_context.get("device_info", {}),
            referrer_source=session_context.get("referrer"),
            last_page_visited=session_context.get("current_page")
        )

        db.add(profile)
        await db.commit()
        await db.refresh(profile)

        return profile

    async def _update_customer_profile(
        self,
        profile: CustomerProfile,
        interaction_data: Dict[str, Any],
        db: AsyncSession
    ) -> CustomerProfile:
        """Update existing customer profile with new interaction data"""

        # Update engagement metrics
        profile.update_engagement_metrics(
            message_count=interaction_data.get("message_count", 1)
        )

        # Update interests if new ones discovered
        for interest in interaction_data.get("interests", []):
            profile.add_interest(interest)

        # Update pain points
        for pain_point in interaction_data.get("pain_points", []):
            profile.add_pain_point(pain_point)

        # Update preferences
        for key, value in interaction_data.get("preferences", {}).items():
            profile.set_preference(key, value)

        await db.commit()
        await db.refresh(profile)

        return profile


# Global service instance
customer_data_service = CustomerDataService()
