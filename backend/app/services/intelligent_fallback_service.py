"""
Intelligent Fallback Service - Provides smart strategies for handling users with minimal context
while maintaining excellent customer experience.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .customer_data_service import CustomerContext, ProductContext


class FallbackStrategy(str, Enum):
    """Different fallback strategies based on context level"""
    DISCOVERY_MODE = "discovery"      # Ask smart questions to learn about user
    ASSUMPTION_MODE = "assumption"     # Make intelligent assumptions
    EXPLORATION_MODE = "exploration"   # Guide user through options
    SHOWCASE_MODE = "showcase"        # Show best offerings proactively
    HELPFUL_MODE = "helpful"          # Focus on being immediately helpful


@dataclass
class FallbackResponse:
    """Enhanced response with fallback strategy applied"""
    strategy_used: FallbackStrategy
    enhanced_prompt: str
    conversation_starters: List[str]
    smart_questions: List[str]
    assumptions_made: Dict[str, Any]
    confidence_level: float


class IntelligentFallbackService:
    """Provides intelligent strategies for minimal context situations"""

    def __init__(self):
        self.intent_patterns = {
            "pricing": [r"\bprice", r"\bcost", r"\bhow much", r"\$", r"expensive", r"cheap", r"budget"],
            "product_info": [r"\bproduct", r"\bfeature", r"\bspec", r"\bdetail", r"\btell me about"],
            "comparison": [r"\bcompare", r"\bvs", r"\bdifference", r"\bbetter", r"\bwhich"],
            "support": [r"\bhelp", r"\bproblem", r"\bissue", r"\bbroken", r"\bnot working"],
            "purchase": [r"\bbuy", r"\border", r"\bpurchase", r"\bget", r"\bneed"],
            "general": [r"\bhello", r"\bhi", r"\binfo", r"\bquestion"]
        }

        self.smart_questions = {
            "discovery": [
                "What brings you here today? I'd love to help you find exactly what you're looking for!",
                "Are you browsing for something specific, or would you like me to show you our most popular options?",
                "Is this for personal use or business? That helps me recommend the best solutions!",
                "What's your main priority - quality, value, or specific features?"
            ],
            "product_focused": [
                "What type of product interests you most?",
                "Are you looking for something specific or exploring your options?",
                "What's most important to you in terms of features?",
                "Do you have a particular budget range in mind?"
            ],
            "support_focused": [
                "What can I help you with today?",
                "Are you experiencing any specific issues?",
                "Would you like help finding information or solving a problem?",
                "How can I make your experience better?"
            ]
        }

        self.assumptions = {
            "new_visitor": {
                "journey_stage": "awareness",
                "information_need": "high",
                "decision_confidence": "low",
                "preferred_approach": "educational"
            },
            "returning_visitor": {
                "journey_stage": "consideration",
                "information_need": "medium",
                "decision_confidence": "medium",
                "preferred_approach": "solution_focused"
            },
            "urgent_visitor": {
                "journey_stage": "decision",
                "information_need": "low",
                "decision_confidence": "high",
                "preferred_approach": "direct"
            }
        }

    def determine_fallback_strategy(
        self,
        customer_context: CustomerContext,
        product_context: ProductContext,
        user_message: str
    ) -> FallbackStrategy:
        """Determine the best fallback strategy based on available context"""

        # Analyze user message for intent
        intent = self._detect_intent(user_message)

        # Factor in confidence level
        confidence = customer_context.confidence_score

        # Consider visitor type
        is_new = not customer_context.is_returning
        has_email = customer_context.email is not None

        # Determine strategy
        if confidence < 0.2:
            return FallbackStrategy.DISCOVERY_MODE
        elif intent == "support" or "problem" in user_message.lower():
            return FallbackStrategy.HELPFUL_MODE
        elif intent in ["pricing", "purchase"] and confidence > 0.3:
            return FallbackStrategy.ASSUMPTION_MODE
        elif is_new and intent == "general":
            return FallbackStrategy.SHOWCASE_MODE
        else:
            return FallbackStrategy.EXPLORATION_MODE

    def apply_fallback_strategy(
        self,
        strategy: FallbackStrategy,
        customer_context: CustomerContext,
        product_context: ProductContext,
        user_message: str,
        base_response: str
    ) -> FallbackResponse:
        """Apply the chosen fallback strategy to enhance the response"""

        if strategy == FallbackStrategy.DISCOVERY_MODE:
            return self._apply_discovery_mode(customer_context, product_context, user_message, base_response)
        elif strategy == FallbackStrategy.ASSUMPTION_MODE:
            return self._apply_assumption_mode(customer_context, product_context, user_message, base_response)
        elif strategy == FallbackStrategy.EXPLORATION_MODE:
            return self._apply_exploration_mode(customer_context, product_context, user_message, base_response)
        elif strategy == FallbackStrategy.SHOWCASE_MODE:
            return self._apply_showcase_mode(customer_context, product_context, user_message, base_response)
        elif strategy == FallbackStrategy.HELPFUL_MODE:
            return self._apply_helpful_mode(customer_context, product_context, user_message, base_response)
        else:
            return self._apply_default_strategy(customer_context, product_context, user_message, base_response)

    def _apply_discovery_mode(self, customer_context, product_context, user_message, base_response) -> FallbackResponse:
        """Discovery mode - ask smart questions to learn about the user"""

        # Choose appropriate discovery questions based on intent
        intent = self._detect_intent(user_message)
        if intent == "support":
            questions = self.smart_questions["support_focused"]
            conversation_approach = "They seem to need help with something specific. Be supportive and solution-focused while discovering their exact needs."
        elif intent in ["product_info", "pricing", "purchase"]:
            questions = self.smart_questions["product_focused"]
            conversation_approach = "They're interested in our products. Show enthusiasm while learning about their specific needs and preferences."
        else:
            questions = self.smart_questions["discovery"]
            conversation_approach = "They're exploring. Create a welcoming experience while naturally discovering what brought them here."

        return FallbackResponse(
            strategy_used=FallbackStrategy.DISCOVERY_MODE,
            enhanced_prompt=conversation_approach,
            conversation_starters=questions[:3],
            smart_questions=questions[:2],  # Limit to most important questions
            assumptions_made=self.assumptions["new_visitor"],
            confidence_level=0.6
        )

    def _apply_assumption_mode(self, customer_context, product_context, user_message, base_response) -> FallbackResponse:
        """Assumption mode - make intelligent assumptions based on context"""

        # Determine what assumptions to make
        assumptions = {}
        conversation_guidance = ""

        if "price" in user_message.lower():
            assumptions["intent"] = "price_conscious"
            assumptions["journey_stage"] = "consideration"
            conversation_guidance = "They're asking about pricing, so they're likely comparing options. Be confident in sharing pricing information and value propositions. Mention popular packages and be ready to discuss value."
        elif "help" in user_message.lower() or "problem" in user_message.lower():
            assumptions["intent"] = "needs_support"
            conversation_guidance = "They need help with something. Be solution-focused and supportive. Ask clarifying questions to understand the specific issue."
        else:
            # Use session context for assumptions
            if customer_context.session_context:
                page = customer_context.session_context.get("current_page", "")
                if "/products" in page:
                    assumptions["category_interest"] = "products"
                    conversation_guidance = "They're browsing products. Show enthusiasm about our offerings and help them explore options that match their needs."
                elif "/pricing" in page:
                    assumptions["intent"] = "pricing_research"
                    conversation_guidance = "They're researching pricing. Be transparent about costs and focus on value. Share popular options and be prepared to discuss different packages."
                else:
                    conversation_guidance = "Make educated assumptions based on their questions. Be confident but ready to adjust if your assumptions don't match their actual needs."
            else:
                conversation_guidance = "Make educated assumptions based on their questions. Be confident but ready to adjust if your assumptions don't match their actual needs."

        return FallbackResponse(
            strategy_used=FallbackStrategy.ASSUMPTION_MODE,
            enhanced_prompt=conversation_guidance,
            conversation_starters=[],
            smart_questions=["Does that match what you're looking for?", "Am I on the right track with what you need?"],
            assumptions_made=assumptions,
            confidence_level=0.7
        )

    def _apply_exploration_mode(self, customer_context, product_context, user_message, base_response) -> FallbackResponse:
        """Exploration mode - guide user through options"""

        # Create exploration paths based on available products
        if product_context and product_context.product_catalog:
            categories = list(set([p.get("category", "General") for p in product_context.product_catalog]))
            exploration_options = categories[:3]
            conversation_guidance = f"Help them explore our {', '.join(categories[:2])} and {categories[2] if len(categories) > 2 else ''} options. Present choices clearly and highlight what's popular or recommended. Make it easy for them to navigate their options."
        else:
            exploration_options = ["Products", "Pricing", "Support", "General Information"]
            conversation_guidance = "Guide them through our main areas: products, pricing, and support. Present clear options and help them find what they're most interested in. Offer to recommend popular choices."

        return FallbackResponse(
            strategy_used=FallbackStrategy.EXPLORATION_MODE,
            enhanced_prompt=conversation_guidance,
            conversation_starters=exploration_options,
            smart_questions=["What interests you most?", "Would you like me to recommend our most popular option?"],
            assumptions_made={"needs_guidance": True},
            confidence_level=0.5
        )

    def _apply_showcase_mode(self, customer_context, product_context, user_message, base_response) -> FallbackResponse:
        """Showcase mode - proactively show best offerings"""

        # Identify best offerings to showcase
        showcases = []
        conversation_guidance = ""

        if product_context:
            if product_context.featured_products:
                featured_names = [p['name'] for p in product_context.featured_products[:2]]
                showcases.extend(featured_names)
                conversation_guidance = f"Lead with enthusiasm about our featured offerings: {', '.join(featured_names)}. "

            if product_context.promotions:
                promo_titles = [p['title'] for p in product_context.promotions[:1]]
                showcases.extend(promo_titles)
                conversation_guidance += f"Mention our current promotion: {promo_titles[0] if promo_titles else ''}. "

        if not conversation_guidance:
            conversation_guidance = "Show enthusiasm about our most popular solutions. "

        conversation_guidance += "Create excitement and value. Make it easy for them to learn more while staying genuinely helpful."

        return FallbackResponse(
            strategy_used=FallbackStrategy.SHOWCASE_MODE,
            enhanced_prompt=conversation_guidance,
            conversation_starters=showcases if showcases else ["Most popular options", "Featured solutions", "Special offers"],
            smart_questions=["What catches your interest?", "Would you like to hear about our most popular choice?"],
            assumptions_made={"intent": "browsing", "openness": "high"},
            confidence_level=0.4
        )

    def _apply_helpful_mode(self, customer_context, product_context, user_message, base_response) -> FallbackResponse:
        """Helpful mode - focus on immediate assistance"""

        conversation_guidance = "Focus on being immediately useful. Listen carefully to their needs and provide clear, actionable help. Be solution-oriented and follow up to make sure they have everything they need. Ask clarifying questions if needed to give the best possible assistance."

        return FallbackResponse(
            strategy_used=FallbackStrategy.HELPFUL_MODE,
            enhanced_prompt=conversation_guidance,
            conversation_starters=["Direct answers", "Step-by-step guidance", "Additional resources"],
            smart_questions=["Is there anything specific I can help clarify?", "What would be most helpful right now?"],
            assumptions_made={"priority": "immediate_help"},
            confidence_level=0.8
        )

    def _apply_default_strategy(self, customer_context, product_context, user_message, base_response) -> FallbackResponse:
        """Default fallback strategy"""

        conversation_guidance = "Be warm, professional, and adaptable. Gather context naturally through conversation while being genuinely helpful. Match their energy and communication style."

        return FallbackResponse(
            strategy_used=FallbackStrategy.HELPFUL_MODE,
            enhanced_prompt=conversation_guidance,
            conversation_starters=[],
            smart_questions=["How can I best help you today?"],
            assumptions_made={},
            confidence_level=0.5
        )

    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent

        return "general"

    def create_context_enriched_prompt(
        self,
        base_system_prompt: str,
        customer_context: CustomerContext,
        product_context: ProductContext,
        fallback_response: FallbackResponse
    ) -> str:
        """Create a comprehensive prompt enriched with all available context"""

        # Start with a natural, conversational foundation
        enhanced_prompt = base_system_prompt

        # Add customer understanding naturally
        customer_insight = self._build_customer_insight(customer_context)
        if customer_insight:
            enhanced_prompt += f"\n\n{customer_insight}"

        # Add business context naturally
        business_context = self._build_business_context(product_context)
        if business_context:
            enhanced_prompt += f"\n\n{business_context}"

        # Add conversation strategy guidance
        strategy_guidance = self._build_strategy_guidance(fallback_response, customer_context)
        enhanced_prompt += f"\n\n{strategy_guidance}"

        # Add core conversation principles
        enhanced_prompt += f"""

CONVERSATION PRINCIPLES:
• Be genuinely helpful and create a positive experience
• Match the customer's energy and communication style
• Use the company information to provide accurate, relevant answers
• When unsure about something, be honest while staying helpful
• Focus on understanding what the customer really needs
• Make the conversation feel natural and engaging

Remember: Great customer service means being genuinely interested in helping, not just providing information."""

        return enhanced_prompt

    def _build_customer_insight(self, customer_context: CustomerContext) -> str:
        """Build natural customer insight section"""
        if customer_context.confidence_score < 0.3:
            return "CUSTOMER CONTEXT: This appears to be a new visitor. Focus on discovery and creating a welcoming first impression."

        insights = []

        if customer_context.is_returning:
            insights.append("This is a returning customer")
        else:
            insights.append("This appears to be a new customer")

        if customer_context.current_interests:
            interests_text = ', '.join(customer_context.current_interests)
            insights.append(f"interested in {interests_text}")

        if customer_context.journey_stage != "awareness":
            insights.append(f"currently in the {customer_context.journey_stage} stage")

        if customer_context.communication_style and customer_context.communication_style != "neutral":
            insights.append(f"prefers {customer_context.communication_style} communication")

        if insights:
            return f"CUSTOMER CONTEXT: {' and '.join(insights)}. Tailor your approach accordingly."

        return ""

    def _build_business_context(self, product_context: ProductContext) -> str:
        """Build natural business context section"""
        if not product_context or not product_context.company_info:
            return ""

        company = product_context.company_info
        context_parts = []

        if company.get('name'):
            context_parts.append(f"You represent {company['name']}")

        if company.get('mission'):
            context_parts.append(f"Our mission: {company['mission']}")

        if company.get('specialties'):
            specialties = ', '.join(company['specialties'])
            context_parts.append(f"We specialize in {specialties}")

        # Add key business policies if available
        if product_context.policies:
            policy_highlights = []
            if product_context.policies.get('return_policy'):
                policy_highlights.append(f"Returns: {product_context.policies['return_policy']}")
            if product_context.policies.get('shipping'):
                policy_highlights.append(f"Shipping: {product_context.policies['shipping']}")

            if policy_highlights:
                context_parts.append(f"Key policies: {' | '.join(policy_highlights)}")

        if context_parts:
            return f"BUSINESS CONTEXT: {'. '.join(context_parts)}."

        return ""

    def _build_strategy_guidance(self, fallback_response: FallbackResponse, customer_context: CustomerContext) -> str:
        """Build natural strategy guidance"""
        strategy = fallback_response.strategy_used
        confidence = customer_context.confidence_score

        if strategy == FallbackStrategy.DISCOVERY_MODE:
            return """APPROACH: Since we don't know much about this customer yet, focus on discovery. Ask thoughtful questions to understand their needs while being genuinely helpful. Make them feel welcome and valued."""

        elif strategy == FallbackStrategy.ASSUMPTION_MODE:
            assumptions_text = ', '.join([f"{k}: {v}" for k, v in fallback_response.assumptions_made.items()])
            return f"""APPROACH: Based on context clues, we're assuming: {assumptions_text}. Proceed confidently with these assumptions while staying open to correction. Verify important assumptions naturally in conversation."""

        elif strategy == FallbackStrategy.EXPLORATION_MODE:
            return """APPROACH: Guide this customer through their options in an organized, helpful way. Present choices clearly and make it easy for them to find what they're looking for. Highlight popular or recommended options."""

        elif strategy == FallbackStrategy.SHOWCASE_MODE:
            return """APPROACH: This is a great opportunity to showcase our best offerings! Lead with enthusiasm and value. Help them discover options they might not have considered while addressing their specific interests."""

        elif strategy == FallbackStrategy.HELPFUL_MODE:
            return """APPROACH: Focus on being immediately useful and solving their problem. Listen carefully, provide clear actionable help, and make sure they feel supported. Follow up to ensure their needs are met."""

        else:
            return f"""APPROACH: Adapt your communication style based on context confidence level ({confidence:.1f}). Be warm, professional, and ready to adjust based on how the conversation develops."""


# Global service instance
intelligent_fallback_service = IntelligentFallbackService()