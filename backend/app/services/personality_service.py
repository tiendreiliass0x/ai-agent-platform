"""
Personality Injection Service for Natural & Engaging Agent Conversations
"""

import json
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..models.persona import Persona
from ..models.agent import Agent
from ..core.database import get_db
from .conversation_enhancer import conversation_enhancer


class PersonalityService:
    def __init__(self):
        self.personality_templates = {
            "sales_rep": {
                "greeting_patterns": [
                    "Hey there! 👋 Great to meet you!",
                    "Hello! How can I help you find something amazing today?",
                    "Hi! I'm excited to help you discover the perfect solution!",
                    "Welcome! Let's find exactly what you're looking for!"
                ],
                "conversation_enhancers": {
                    "enthusiasm": 0.8,
                    "empathy": 0.7,
                    "confidence": 0.9,
                    "helpfulness": 0.9
                },
                "response_modifiers": {
                    "use_emojis": True,
                    "ask_followups": True,
                    "show_excitement": True,
                    "personal_touch": True
                }
            },
            "solution_engineer": {
                "greeting_patterns": [
                    "Hi! I'm here to help you build the perfect solution.",
                    "Hello! Let's dive into the technical details together.",
                    "Hey! Ready to explore some innovative approaches?",
                    "Hi there! I love solving complex challenges - what's on your mind?"
                ],
                "conversation_enhancers": {
                    "technical_depth": 0.9,
                    "problem_solving": 0.9,
                    "creativity": 0.8,
                    "precision": 0.8
                },
                "response_modifiers": {
                    "use_emojis": False,
                    "technical_language": True,
                    "provide_alternatives": True,
                    "detail_oriented": True
                }
            },
            "support_expert": {
                "greeting_patterns": [
                    "Hi! I'm here to help you get everything working smoothly.",
                    "Hello! Let's resolve this together step by step.",
                    "Hey there! No worries - I've got you covered.",
                    "Hi! I understand this can be frustrating - let me help!"
                ],
                "conversation_enhancers": {
                    "patience": 0.9,
                    "empathy": 0.9,
                    "clarity": 0.8,
                    "reassurance": 0.8
                },
                "response_modifiers": {
                    "use_emojis": False,
                    "step_by_step": True,
                    "reassuring_tone": True,
                    "check_understanding": True
                }
            },
            "domain_specialist": {
                "greeting_patterns": [
                    "Hello! I'm delighted to share my expertise with you.",
                    "Hi! Let's explore the nuances of this topic together.",
                    "Hey! I love discussing the finer details - what interests you?",
                    "Hello! My specialty is making complex topics accessible."
                ],
                "conversation_enhancers": {
                    "expertise": 0.9,
                    "depth": 0.8,
                    "educational": 0.8,
                    "thoughtfulness": 0.9
                },
                "response_modifiers": {
                    "use_emojis": False,
                    "provide_context": True,
                    "educational_tone": True,
                    "cite_expertise": True
                }
            }
        }

    async def get_agent_personality(self, agent_id: int, db: Session) -> Dict[str, Any]:
        """Retrieve agent's personality configuration"""
        try:
            # Get agent with persona relationship
            agent = await db.execute(
                select(Agent).where(Agent.id == agent_id)
                .joinedload(Agent.persona)
            )
            agent = agent.scalar_one_or_none()

            if not agent:
                return self._get_default_personality()

            if agent.persona:
                # Use persona-based personality
                return self._build_personality_from_persona(agent.persona)
            else:
                # Use default friendly assistant personality
                return self._get_default_personality()

        except Exception as e:
            print(f"Error getting agent personality: {e}")
            return self._get_default_personality()

    def _build_personality_from_persona(self, persona: Persona) -> Dict[str, Any]:
        """Build personality configuration from persona data"""
        template_name = persona.template_name or "domain_specialist"
        base_template = self.personality_templates.get(template_name, self.personality_templates["domain_specialist"])

        # Merge persona-specific data with base template
        personality = {
            "name": persona.name,
            "description": persona.description,
            "system_prompt": persona.system_prompt,
            "template_name": template_name,
            **base_template,
            "tactics": persona.tactics or {},
            "communication_style": persona.communication_style or {},
            "response_patterns": persona.response_patterns or {}
        }

        return personality

    def _get_default_personality(self) -> Dict[str, Any]:
        """Default friendly assistant personality"""
        return {
            "name": "Helpful Assistant",
            "description": "A friendly, knowledgeable assistant",
            "template_name": "domain_specialist",
            "system_prompt": "You are a helpful, knowledgeable, and friendly assistant.",
            **self.personality_templates["domain_specialist"]
        }

    def inject_personality_into_prompt(self, base_prompt: str, personality: Dict[str, Any], user_query: str, context: str = "") -> str:
        """Enhance system prompt with personality traits"""

        personality_enhancement = f"""
PERSONALITY PROFILE:
- Role: {personality.get('name', 'Assistant')}
- Style: {personality.get('description', 'Helpful and knowledgeable')}

CONVERSATION GUIDELINES:
{self._generate_conversation_guidelines(personality)}

RESPONSE ENHANCEMENT:
{self._generate_response_enhancement_rules(personality)}

ORIGINAL CONTEXT:
{context}

ORIGINAL INSTRUCTIONS:
{base_prompt}

Remember: Be natural, engaging, and authentic. Your personality should enhance the conversation, not overshadow the helpful information you provide.
"""
        return personality_enhancement

    def _generate_conversation_guidelines(self, personality: Dict[str, Any]) -> str:
        """Generate conversation guidelines based on personality"""
        template_name = personality.get("template_name", "domain_specialist")
        enhancers = personality.get("conversation_enhancers", {})

        guidelines = []

        if template_name == "sales_rep":
            guidelines.extend([
                "- Be enthusiastic and positive about helping users find solutions",
                "- Show genuine excitement about products and possibilities",
                "- Ask thoughtful follow-up questions to understand needs better",
                "- Use a warm, friendly tone that builds trust and rapport"
            ])

        elif template_name == "solution_engineer":
            guidelines.extend([
                "- Focus on technical accuracy and practical solutions",
                "- Provide clear, logical explanations of complex concepts",
                "- Offer alternative approaches when appropriate",
                "- Balance technical depth with accessibility"
            ])

        elif template_name == "support_expert":
            guidelines.extend([
                "- Be patient and reassuring, especially with frustrated users",
                "- Break down solutions into clear, manageable steps",
                "- Check for understanding before moving to next steps",
                "- Maintain a calm, helpful demeanor throughout"
            ])

        else:  # domain_specialist
            guidelines.extend([
                "- Share knowledge with genuine enthusiasm for the subject",
                "- Provide context and background to help users understand",
                "- Be thorough but not overwhelming in explanations",
                "- Encourage further questions and exploration"
            ])

        # Add enhancer-specific guidelines
        if enhancers.get("empathy", 0) > 0.7:
            guidelines.append("- Show understanding and validation of user concerns")
        if enhancers.get("enthusiasm", 0) > 0.7:
            guidelines.append("- Express genuine excitement about helping and solving problems")

        return '\n'.join(guidelines)

    def _generate_response_enhancement_rules(self, personality: Dict[str, Any]) -> str:
        """Generate response enhancement rules"""
        modifiers = personality.get("response_modifiers", {})

        rules = []

        if modifiers.get("use_emojis", False):
            rules.append("- Use appropriate emojis sparingly to add warmth (1-2 per response max)")
        if modifiers.get("ask_followups", False):
            rules.append("- End responses with relevant follow-up questions when appropriate")
        if modifiers.get("personal_touch", False):
            rules.append("- Add personal touches that show you care about the user's success")
        if modifiers.get("step_by_step", False):
            rules.append("- Break complex information into numbered steps when helpful")
        if modifiers.get("provide_alternatives", False):
            rules.append("- Offer multiple solutions or approaches when relevant")
        if modifiers.get("reassuring_tone", False):
            rules.append("- Use reassuring language that builds confidence")
        if modifiers.get("check_understanding", False):
            rules.append("- Ask for confirmation that explanations make sense")

        # Default enhancement rules
        rules.extend([
            "- Keep responses conversational and engaging",
            "- Match the user's energy level appropriately",
            "- Be concise but thorough - no unnecessary verbosity",
            "- End with a positive, helpful tone"
        ])

        return '\n'.join(rules)

    def enhance_response_with_personality(self, response: str, personality: Dict[str, Any], user_query: str, conversation_history: List[Dict] = None) -> str:
        """Post-process response to add personality touches and natural conversation flow"""

        template_name = personality.get("template_name", "domain_specialist")
        modifiers = personality.get("response_modifiers", {})

        # Step 1: Apply conversation enhancement for natural flow
        enhanced_response = conversation_enhancer.enhance_response(
            response=response,
            user_message=user_query,
            conversation_history=conversation_history or [],
            personality_type=template_name,
            add_good_vibes=modifiers.get("use_emojis", False) or template_name == "sales_rep"
        )

        # Step 2: Add personality-specific flavor
        user_mood = conversation_enhancer.detect_user_mood(user_query, conversation_history)
        enhanced_response = conversation_enhancer.add_personality_flavor(
            response=enhanced_response,
            personality_type=template_name,
            user_mood=user_mood
        )

        # Step 3: Apply specific personality modifiers
        if template_name == "sales_rep" and modifiers.get("show_excitement", False):
            if "!" not in enhanced_response[-20:]:  # If no excitement in ending
                enhanced_response = enhanced_response.rstrip('.') + "!"

        # Step 4: Add greetings for first interactions
        if any(greeting in user_query.lower() for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']):
            greetings = personality.get("greeting_patterns", [])
            if greetings and not any(greeting.lower() in enhanced_response.lower() for greeting in ['hi', 'hello', 'hey']):
                # Only add greeting if conversation enhancer didn't already add one
                if not enhanced_response.startswith(('Hi', 'Hey', 'Hello')):
                    selected_greeting = greetings[0]
                    enhanced_response = f"{selected_greeting}\n\n{enhanced_response}"

        return enhanced_response

    async def get_conversation_context_hints(self, agent_id: int, conversation_history: List[Dict], db: Session) -> Dict[str, Any]:
        """Analyze conversation for personality context hints"""

        personality = await self.get_agent_personality(agent_id, db)

        # Analyze conversation patterns
        context_hints = {
            "user_mood": self._detect_user_mood(conversation_history),
            "conversation_stage": self._detect_conversation_stage(conversation_history),
            "topics_discussed": self._extract_topics(conversation_history),
            "personality_adjustments": self._suggest_personality_adjustments(personality, conversation_history)
        }

        return context_hints

    def _detect_user_mood(self, history: List[Dict]) -> str:
        """Detect user's mood from conversation history"""
        if not history:
            return "neutral"

        recent_messages = [msg for msg in history[-3:] if msg.get("role") == "user"]

        frustration_indicators = ['not working', 'broken', 'frustrated', 'help!', 'urgent', 'problem']
        excitement_indicators = ['great', 'awesome', 'love', 'perfect', 'amazing', '!']

        text = ' '.join([msg.get("content", "").lower() for msg in recent_messages])

        if any(indicator in text for indicator in frustration_indicators):
            return "frustrated"
        elif any(indicator in text for indicator in excitement_indicators):
            return "excited"
        else:
            return "neutral"

    def _detect_conversation_stage(self, history: List[Dict]) -> str:
        """Detect what stage of conversation we're in"""
        if len(history) <= 2:
            return "greeting"
        elif len(history) <= 6:
            return "exploration"
        else:
            return "deep_discussion"

    def _extract_topics(self, history: List[Dict]) -> List[str]:
        """Extract main topics from conversation"""
        # Simple keyword extraction - could be enhanced with NLP
        user_messages = [msg.get("content", "").lower() for msg in history if msg.get("role") == "user"]
        text = ' '.join(user_messages)

        # Basic topic keywords - this could be much more sophisticated
        topic_keywords = {
            "product": ["product", "item", "buy", "purchase", "price"],
            "technical": ["how", "setup", "configure", "install", "api"],
            "support": ["issue", "problem", "error", "broken", "help"],
            "information": ["what", "why", "when", "where", "explain"]
        }

        topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)

        return topics

    def _suggest_personality_adjustments(self, personality: Dict[str, Any], history: List[Dict]) -> Dict[str, Any]:
        """Suggest personality adjustments based on conversation flow"""

        user_mood = self._detect_user_mood(history)
        conversation_stage = self._detect_conversation_stage(history)

        adjustments = {}

        if user_mood == "frustrated":
            adjustments.update({
                "increase_empathy": True,
                "use_reassuring_tone": True,
                "be_more_patient": True,
                "avoid_emojis": True
            })
        elif user_mood == "excited":
            adjustments.update({
                "match_enthusiasm": True,
                "use_positive_language": True,
                "show_excitement": True
            })

        if conversation_stage == "greeting":
            adjustments.update({
                "warm_welcome": True,
                "set_helpful_tone": True
            })
        elif conversation_stage == "deep_discussion":
            adjustments.update({
                "maintain_focus": True,
                "provide_depth": True,
                "check_comprehension": True
            })

        return adjustments


# Global personality service instance
personality_service = PersonalityService()