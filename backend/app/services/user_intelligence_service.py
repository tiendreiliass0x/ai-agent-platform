"""
Advanced User Intelligence Service
Real-time user behavior analysis and intent prediction for world-class concierge experience.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.services.gemini_service import gemini_service
from app.services.memory_service import memory_service


class UrgencyLevel(str, Enum):
    """User urgency detection"""
    CRITICAL = "critical"      # Immediate attention needed
    HIGH = "high"             # Priority response required
    MEDIUM = "medium"         # Standard response time
    LOW = "low"              # Can be queued


class EmotionalState(str, Enum):
    """Detected emotional states"""
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    EXCITED = "excited"
    SATISFIED = "satisfied"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    HAPPY = "happy"
    WORRIED = "worried"


class IntentCategory(str, Enum):
    """User intent categories"""
    INFORMATION_SEEKING = "information_seeking"
    PROBLEM_SOLVING = "problem_solving"
    PURCHASE_INTENT = "purchase_intent"
    SUPPORT_REQUEST = "support_request"
    COMPLAINT = "complaint"
    COMPARISON = "comparison"
    EXPLORATION = "exploration"
    CANCELLATION = "cancellation"


@dataclass
class UserInsight:
    """Real-time user insight"""
    insight_type: str
    confidence: float
    value: Any
    evidence: List[str]
    urgency: UrgencyLevel
    timestamp: datetime


@dataclass
class ConversationAnalysis:
    """Comprehensive conversation analysis"""
    emotional_state: EmotionalState
    urgency_level: UrgencyLevel
    intent_category: IntentCategory
    confidence_score: float
    key_topics: List[str]
    pain_points: List[str]
    opportunities: List[str]
    next_best_actions: List[str]
    personalization_cues: Dict[str, Any]


class UserIntelligenceService:
    """Advanced user intelligence and behavior analysis"""

    def __init__(self):
        self.urgency_keywords = {
            UrgencyLevel.CRITICAL: [
                "urgent", "emergency", "asap", "immediately", "critical", "crisis",
                "broken", "down", "not working", "crashed", "lost data", "security breach"
            ],
            UrgencyLevel.HIGH: [
                "quickly", "soon", "priority", "important", "deadline", "meeting",
                "client", "customer", "presentation", "tomorrow"
            ],
            UrgencyLevel.MEDIUM: [
                "help", "question", "issue", "problem", "how to", "need"
            ]
        }

        self.emotional_indicators = {
            EmotionalState.FRUSTRATED: [
                "frustrated", "annoyed", "difficult", "complicated", "confusing",
                "waste of time", "doesn't work", "terrible", "awful"
            ],
            EmotionalState.ANGRY: [
                "angry", "mad", "furious", "ridiculous", "unacceptable",
                "terrible service", "disappointed", "outrageous"
            ],
            EmotionalState.EXCITED: [
                "excited", "amazing", "awesome", "fantastic", "love it",
                "can't wait", "perfect", "exactly what I need"
            ],
            EmotionalState.CONFUSED: [
                "confused", "don't understand", "unclear", "what does this mean",
                "how does this work", "I'm lost", "doesn't make sense"
            ]
        }

    async def analyze_user_message(
        self,
        message: str,
        customer_profile_id: int,
        conversation_history: List[Dict[str, str]] = None,
        session_context: Dict[str, Any] = None
    ) -> ConversationAnalysis:
        """Comprehensive real-time message analysis"""

        # Run analysis in parallel for speed
        analysis_tasks = [
            self._detect_emotional_state(message, conversation_history),
            self._detect_urgency_level(message, session_context),
            self._predict_intent(message, conversation_history),
            self._extract_key_topics(message),
            self._identify_pain_points(message),
            self._spot_opportunities(message, conversation_history),
            self._suggest_next_actions(message, customer_profile_id),
            self._extract_personalization_cues(message, session_context)
        ]

        results = await asyncio.gather(*analysis_tasks)

        return ConversationAnalysis(
            emotional_state=results[0],
            urgency_level=results[1],
            intent_category=results[2],
            confidence_score=self._calculate_overall_confidence(results),
            key_topics=results[3],
            pain_points=results[4],
            opportunities=results[5],
            next_best_actions=results[6],
            personalization_cues=results[7]
        )

    async def _detect_emotional_state(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> EmotionalState:
        """Detect user's emotional state using AI and keywords"""

        # Quick keyword-based detection
        message_lower = message.lower()
        for emotion, keywords in self.emotional_indicators.items():
            if any(keyword in message_lower for keyword in keywords):
                return emotion

        # AI-powered sentiment analysis for nuanced detection
        try:
            context = ""
            if conversation_history:
                context = "Previous conversation: " + " ".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in conversation_history[-3:]
                ])

            analysis_prompt = f"""
Analyze the emotional state in this message. Consider context if provided.

Message: "{message}"
{context}

Classify the emotion as one of: frustrated, confused, excited, satisfied, angry, neutral, happy, worried

Respond with just the emotion word.
"""

            emotion_result = await gemini_service.generate_response(
                prompt=analysis_prompt,
                temperature=0.1,
                max_tokens=50
            )

            detected_emotion = emotion_result.strip().lower()
            if detected_emotion in [e.value for e in EmotionalState]:
                return EmotionalState(detected_emotion)

        except Exception as e:
            print(f"Error in emotion detection: {e}")

        return EmotionalState.NEUTRAL

    async def _detect_urgency_level(
        self,
        message: str,
        session_context: Dict[str, Any] = None
    ) -> UrgencyLevel:
        """Detect urgency level from message content and context"""

        message_lower = message.lower()

        # Check for explicit urgency keywords
        for urgency, keywords in self.urgency_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return urgency

        # Context-based urgency detection
        if session_context:
            # Check for urgent page contexts
            page_url = session_context.get('page_url', '').lower()
            if any(indicator in page_url for indicator in ['support', 'help', 'error', 'issue']):
                return UrgencyLevel.HIGH

            # Check time-based urgency (business hours vs off-hours)
            current_hour = datetime.now().hour
            if current_hour < 9 or current_hour > 17:  # Off-hours = higher urgency
                return UrgencyLevel.HIGH

        # Message length and punctuation patterns
        if '!!!' in message or message.isupper() and len(message) > 10:
            return UrgencyLevel.HIGH

        # Question marks indicate need for help
        if message.count('?') > 1:
            return UrgencyLevel.MEDIUM

        return UrgencyLevel.LOW

    async def _predict_intent(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> IntentCategory:
        """Predict user intent using AI analysis"""

        try:
            context = ""
            if conversation_history:
                context = "Conversation context: " + " ".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in conversation_history[-5:]
                ])

            intent_prompt = f"""
Analyze the user's intent in this message. Consider the conversation context.

Message: "{message}"
{context}

Classify the intent as one of:
- information_seeking: Looking for information or answers
- problem_solving: Trying to solve a technical issue
- purchase_intent: Interested in buying or upgrading
- support_request: Need help with existing service
- complaint: Expressing dissatisfaction
- comparison: Comparing options or features
- exploration: General browsing/discovery
- cancellation: Want to cancel or downgrade

Respond with just the intent category.
"""

            intent_result = await gemini_service.generate_response(
                prompt=intent_prompt,
                temperature=0.1,
                max_tokens=50
            )

            detected_intent = intent_result.strip().lower()
            if detected_intent in [i.value for i in IntentCategory]:
                return IntentCategory(detected_intent)

        except Exception as e:
            print(f"Error in intent prediction: {e}")

        # Fallback to keyword-based detection
        message_lower = message.lower()
        if any(word in message_lower for word in ['buy', 'purchase', 'price', 'cost', 'upgrade']):
            return IntentCategory.PURCHASE_INTENT
        elif any(word in message_lower for word in ['help', 'support', 'issue', 'problem']):
            return IntentCategory.SUPPORT_REQUEST
        elif any(word in message_lower for word in ['cancel', 'unsubscribe', 'stop', 'quit']):
            return IntentCategory.CANCELLATION
        elif '?' in message:
            return IntentCategory.INFORMATION_SEEKING

        return IntentCategory.EXPLORATION

    async def _extract_key_topics(self, message: str) -> List[str]:
        """Extract key topics and entities from message"""

        try:
            topics_prompt = f"""
Extract the main topics, features, or concepts mentioned in this message.

Message: "{message}"

List the key topics as a comma-separated list. Focus on business-relevant topics, features, products, or concepts.
Example: "pricing, security, integration, API, deployment"

Respond with just the comma-separated list.
"""

            topics_result = await gemini_service.generate_response(
                prompt=topics_prompt,
                temperature=0.2,
                max_tokens=100
            )

            topics = [topic.strip() for topic in topics_result.split(',') if topic.strip()]
            return topics[:5]  # Limit to top 5 topics

        except Exception as e:
            print(f"Error in topic extraction: {e}")
            return []

    async def _identify_pain_points(self, message: str) -> List[str]:
        """Identify specific pain points mentioned by user"""

        pain_indicators = [
            "difficult", "hard", "complicated", "confusing", "slow", "expensive",
            "doesn't work", "broken", "issue", "problem", "challenge", "struggle",
            "time-consuming", "frustrating", "complex", "unclear"
        ]

        pain_points = []
        message_lower = message.lower()

        for indicator in pain_indicators:
            if indicator in message_lower:
                # Extract context around the pain point
                sentences = message.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        pain_points.append(sentence.strip())
                        break

        return pain_points[:3]  # Limit to top 3

    async def _spot_opportunities(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> List[str]:
        """Identify sales/upsell opportunities"""

        opportunities = []
        message_lower = message.lower()

        # Growth indicators
        growth_signals = [
            "scaling", "growing", "expanding", "more users", "team size",
            "enterprise", "multiple", "integration", "automation"
        ]

        for signal in growth_signals:
            if signal in message_lower:
                opportunities.append(f"Growth opportunity detected: {signal}")

        # Feature interest
        if any(word in message_lower for word in ['api', 'integration', 'custom']):
            opportunities.append("Technical integration interest")

        if any(word in message_lower for word in ['team', 'collaboration', 'share']):
            opportunities.append("Team/collaboration features interest")

        return opportunities

    async def _suggest_next_actions(
        self,
        message: str,
        customer_profile_id: int
    ) -> List[str]:
        """Suggest optimal next actions for the agent"""

        # Get customer profile for personalized actions
        try:
            # This would integrate with memory service to get customer context
            actions = []

            message_lower = message.lower()

            # Immediate actions based on message content
            if any(word in message_lower for word in ['urgent', 'asap', 'emergency']):
                actions.append("Escalate to human agent immediately")

            if '?' in message:
                actions.append("Provide comprehensive answer with examples")

            if any(word in message_lower for word in ['demo', 'trial', 'test']):
                actions.append("Offer product demo or trial")

            if any(word in message_lower for word in ['price', 'cost', 'pricing']):
                actions.append("Share pricing information and schedule consultation")

            if any(word in message_lower for word in ['cancel', 'unsubscribe']):
                actions.append("Understand reasons and offer retention options")

            # Default helpful actions
            if not actions:
                actions = [
                    "Provide detailed helpful response",
                    "Ask clarifying questions if needed",
                    "Offer additional resources"
                ]

            return actions

        except Exception as e:
            print(f"Error generating next actions: {e}")
            return ["Provide helpful response"]

    async def _extract_personalization_cues(
        self,
        message: str,
        session_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract cues for personalizing responses"""

        cues = {}

        # Technical level indicators
        message_lower = message.lower()
        if any(term in message_lower for term in ['api', 'code', 'integration', 'webhook', 'sdk']):
            cues['technical_level'] = 'expert'
        elif any(term in message_lower for term in ['simple', 'easy', 'beginner', 'new to this']):
            cues['technical_level'] = 'beginner'

        # Company size indicators
        if any(term in message_lower for term in ['enterprise', 'corporation', 'large team']):
            cues['company_size'] = 'enterprise'
        elif any(term in message_lower for term in ['startup', 'small team', 'freelancer']):
            cues['company_size'] = 'small'

        # Communication style
        if message.isupper() or '!' in message:
            cues['communication_style'] = 'direct'
        elif len(message.split()) > 50:
            cues['communication_style'] = 'detailed'
        else:
            cues['communication_style'] = 'concise'

        # Time sensitivity
        if any(term in message_lower for term in ['today', 'now', 'immediately', 'deadline']):
            cues['time_sensitive'] = True

        return cues

    def _calculate_overall_confidence(self, analysis_results: List[Any]) -> float:
        """Calculate overall confidence score for the analysis"""

        # Simple confidence calculation based on successful analysis
        successful_analyses = sum(1 for result in analysis_results if result is not None)
        total_analyses = len(analysis_results)

        base_confidence = successful_analyses / total_analyses

        # Boost confidence if multiple indicators align
        return min(0.95, base_confidence * 1.1)

    async def generate_smart_response_strategy(
        self,
        analysis: ConversationAnalysis,
        customer_profile_id: int,
        agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate intelligent response strategy based on analysis"""

        strategy = {
            "response_tone": self._determine_response_tone(analysis),
            "response_length": self._determine_response_length(analysis),
            "personalization_elements": self._get_personalization_elements(analysis),
            "urgency_handling": self._get_urgency_handling(analysis),
            "follow_up_actions": analysis.next_best_actions,
            "escalation_needed": self._should_escalate(analysis)
        }

        return strategy

    def _determine_response_tone(self, analysis: ConversationAnalysis) -> str:
        """Determine appropriate response tone"""

        if analysis.emotional_state == EmotionalState.ANGRY:
            return "empathetic_apologetic"
        elif analysis.emotional_state == EmotionalState.FRUSTRATED:
            return "understanding_solution_focused"
        elif analysis.emotional_state == EmotionalState.EXCITED:
            return "enthusiastic_supportive"
        elif analysis.emotional_state == EmotionalState.CONFUSED:
            return "patient_explanatory"
        else:
            return "professional_friendly"

    def _determine_response_length(self, analysis: ConversationAnalysis) -> str:
        """Determine optimal response length"""

        if analysis.urgency_level == UrgencyLevel.CRITICAL:
            return "brief_actionable"
        elif analysis.emotional_state == EmotionalState.CONFUSED:
            return "detailed_with_examples"
        elif analysis.intent_category == IntentCategory.INFORMATION_SEEKING:
            return "comprehensive"
        else:
            return "medium"

    def _get_personalization_elements(self, analysis: ConversationAnalysis) -> List[str]:
        """Get personalization elements to include"""

        elements = []

        # Add personalization based on cues
        if analysis.personalization_cues.get('technical_level') == 'expert':
            elements.append("include_technical_details")
        elif analysis.personalization_cues.get('technical_level') == 'beginner':
            elements.append("use_simple_language")

        if analysis.personalization_cues.get('time_sensitive'):
            elements.append("prioritize_quick_solutions")

        if analysis.personalization_cues.get('company_size') == 'enterprise':
            elements.append("mention_enterprise_features")

        return elements

    def _get_urgency_handling(self, analysis: ConversationAnalysis) -> Dict[str, Any]:
        """Get urgency handling instructions"""

        if analysis.urgency_level == UrgencyLevel.CRITICAL:
            return {
                "response_priority": "immediate",
                "include_escalation_path": True,
                "response_time_target": "< 30 seconds"
            }
        elif analysis.urgency_level == UrgencyLevel.HIGH:
            return {
                "response_priority": "high",
                "include_contact_info": True,
                "response_time_target": "< 2 minutes"
            }
        else:
            return {
                "response_priority": "normal",
                "response_time_target": "< 5 minutes"
            }

    def _should_escalate(self, analysis: ConversationAnalysis) -> bool:
        """Determine if conversation should be escalated to human"""

        escalation_conditions = [
            analysis.urgency_level == UrgencyLevel.CRITICAL,
            analysis.emotional_state == EmotionalState.ANGRY,
            analysis.intent_category == IntentCategory.CANCELLATION,
            analysis.intent_category == IntentCategory.COMPLAINT
        ]

        return any(escalation_conditions)


# Global instance
user_intelligence_service = UserIntelligenceService()