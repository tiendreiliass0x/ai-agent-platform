"""
Conversation Enhancer - Add natural flow and good vibes to agent responses
"""

import random
from typing import Dict, List, Any, Optional
from datetime import datetime
import re


class ConversationEnhancer:
    def __init__(self):
        # Natural conversation starters
        self.conversation_openers = {
            "greeting": [
                "Hey there! 👋",
                "Hello! Great to meet you!",
                "Hi! How's your day going?",
                "Hey! Ready to dive in?",
                "Hello! I'm excited to help!"
            ],
            "follow_up": [
                "That's a great question!",
                "I love that you're thinking about this!",
                "Awesome, let me help with that!",
                "Perfect timing for this question!",
                "That's exactly what I was hoping you'd ask!"
            ],
            "clarification": [
                "Let me make sure I understand...",
                "Just to clarify...",
                "I want to get this right for you...",
                "Let me break this down...",
                "Here's what I'm hearing..."
            ]
        }

        # Positive reinforcement phrases
        self.positive_reinforcement = [
            "You're on the right track!",
            "That's a smart approach!",
            "Great thinking!",
            "You've got it!",
            "Exactly!",
            "Perfect!",
            "That makes total sense!",
            "Love the way you're approaching this!",
            "You're absolutely right!",
            "Brilliant question!"
        ]

        # Empathetic responses
        self.empathetic_responses = {
            "frustration": [
                "I totally get why that would be frustrating.",
                "That sounds really challenging.",
                "I can understand how that would be annoying.",
                "You're right to be concerned about that.",
                "I hear you - that's definitely not ideal."
            ],
            "confusion": [
                "No worries, this can be tricky!",
                "That's a common point of confusion.",
                "Let me clear that up for you!",
                "I'm here to help make this clearer.",
                "Don't worry, we'll figure this out together!"
            ],
            "excitement": [
                "I love your enthusiasm!",
                "That's awesome!",
                "Your excitement is contagious!",
                "That's fantastic!",
                "I'm excited about this too!"
            ]
        }

        # Conversation connectors
        self.connectors = {
            "building": [
                "Building on that...",
                "Taking it a step further...",
                "Here's another way to think about it...",
                "Additionally...",
                "On top of that..."
            ],
            "contrasting": [
                "On the flip side...",
                "However...",
                "Alternatively...",
                "From another angle...",
                "That said..."
            ],
            "concluding": [
                "To wrap this up...",
                "In summary...",
                "The bottom line is...",
                "Here's the key takeaway...",
                "So in short..."
            ]
        }

        # Encouraging follow-ups
        self.follow_up_encouragers = [
            "What else would you like to know?",
            "Any other questions on your mind?",
            "Is there anything specific you'd like me to elaborate on?",
            "What would be most helpful to explore next?",
            "I'm here if you need anything else!",
            "Feel free to ask about anything else!",
            "What other aspects interest you?",
            "Any other areas you'd like to dive into?"
        ]

        # Good vibes phrases
        self.good_vibes_phrases = [
            "Hope this helps! ✨",
            "You've got this! 💪",
            "Happy to help! 😊",
            "Keep up the great work!",
            "You're doing amazing!",
            "This is going to be great!",
            "I believe in you!",
            "You're on fire today!",
            "Way to go!",
            "You're crushing it!"
        ]

    def detect_user_mood(self, message: str, conversation_history: List[Dict] = None) -> str:
        """Detect user mood from their message"""
        message_lower = message.lower()

        # Frustration indicators
        if any(word in message_lower for word in ['frustrated', 'annoying', 'broken', 'stupid', 'hate', 'terrible', 'awful']):
            return "frustrated"

        # Confusion indicators
        if any(word in message_lower for word in ['confused', 'don\'t understand', 'unclear', 'confusing', 'lost']):
            return "confused"

        # Excitement indicators
        if any(word in message_lower for word in ['awesome', 'amazing', 'love', 'great', 'fantastic', 'perfect']) or '!' in message:
            return "excited"

        # Greeting indicators
        if any(word in message_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']):
            return "greeting"

        return "neutral"

    def is_conversation_starter(self, conversation_history: List[Dict]) -> bool:
        """Check if this is the start of a new conversation"""
        return len(conversation_history) <= 1

    def enhance_response(self,
                        response: str,
                        user_message: str,
                        conversation_history: List[Dict] = None,
                        personality_type: str = "friendly",
                        add_good_vibes: bool = True) -> str:
        """Enhance response with natural conversation flow and good vibes"""

        user_mood = self.detect_user_mood(user_message, conversation_history)
        is_new_conversation = self.is_conversation_starter(conversation_history or [])

        enhanced_response = response

        # Add conversation opener if appropriate
        if is_new_conversation and user_mood == "greeting":
            opener = random.choice(self.conversation_openers["greeting"])
            enhanced_response = f"{opener}\n\n{enhanced_response}"
        elif user_mood in ["confused", "frustrated"]:
            empathetic = random.choice(self.empathetic_responses[user_mood])
            enhanced_response = f"{empathetic} {enhanced_response}"
        elif user_mood == "excited":
            excitement = random.choice(self.empathetic_responses["excitement"])
            enhanced_response = f"{excitement} {enhanced_response}"

        # Add positive reinforcement for good questions
        if self._is_good_question(user_message):
            reinforcement = random.choice(self.positive_reinforcement)
            enhanced_response = f"{reinforcement} {enhanced_response}"

        # Make the response more conversational
        enhanced_response = self._make_conversational(enhanced_response)

        # Add follow-up encouragement
        if not self._ends_with_question(enhanced_response) and len(enhanced_response.split()) > 20:
            follow_up = random.choice(self.follow_up_encouragers)
            enhanced_response = f"{enhanced_response}\n\n{follow_up}"

        # Add good vibes (sparingly)
        if add_good_vibes and personality_type in ["friendly", "sales_rep"] and random.random() < 0.3:
            good_vibes = random.choice(self.good_vibes_phrases)
            enhanced_response = f"{enhanced_response} {good_vibes}"

        return enhanced_response.strip()

    def _is_good_question(self, message: str) -> bool:
        """Check if user asked a particularly good or thoughtful question"""
        thoughtful_indicators = [
            'how does', 'why would', 'what if', 'can you explain',
            'help me understand', 'what\'s the difference', 'best practice'
        ]
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in thoughtful_indicators)

    def _ends_with_question(self, text: str) -> bool:
        """Check if text ends with a question"""
        return text.strip().endswith('?')

    def _make_conversational(self, response: str) -> str:
        """Make response more conversational by adding natural language patterns"""

        # Add conversational connectors between sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)

        if len(sentences) <= 1:
            return response

        enhanced_sentences = [sentences[0]]

        for i, sentence in enumerate(sentences[1:], 1):
            if i < len(sentences) - 1 and random.random() < 0.2:  # Add connectors occasionally
                connector_type = random.choice(["building", "contrasting"])
                if connector_type in self.connectors and self.connectors[connector_type]:
                    connector = random.choice(self.connectors[connector_type])
                    enhanced_sentences.append(f"{connector} {sentence.lower()}")
                else:
                    enhanced_sentences.append(sentence)
            else:
                enhanced_sentences.append(sentence)

        return ' '.join(enhanced_sentences)

    def add_personality_flavor(self, response: str, personality_type: str, user_mood: str) -> str:
        """Add personality-specific flavoring to responses"""

        if personality_type == "sales_rep":
            # Add enthusiasm and benefit-focused language
            if user_mood in ["excited", "neutral"]:
                response = self._add_sales_enthusiasm(response)

        elif personality_type == "support_expert":
            # Add patient, step-by-step language
            response = self._add_support_patience(response)

        elif personality_type == "solution_engineer":
            # Add technical confidence
            response = self._add_technical_precision(response)

        elif personality_type == "domain_specialist":
            # Add educational tone
            response = self._add_educational_depth(response)

        return response

    def _add_sales_enthusiasm(self, response: str) -> str:
        """Add sales enthusiasm to response"""
        enthusiasm_markers = ["This is going to be perfect for", "You're going to love", "This will definitely"]
        if random.random() < 0.3:
            marker = random.choice(enthusiasm_markers)
            # Simple transformation - could be more sophisticated
            response = response.replace("This", marker, 1)
        return response

    def _add_support_patience(self, response: str) -> str:
        """Add patient, supportive tone"""
        patient_starters = ["Let's work through this together.", "No problem at all.", "I'm here to help every step of the way."]
        if not any(starter.lower() in response.lower() for starter in patient_starters):
            starter = random.choice(patient_starters)
            response = f"{starter} {response}"
        return response

    def _add_technical_precision(self, response: str) -> str:
        """Add technical precision markers"""
        precision_phrases = ["Specifically,", "To be precise,", "Here's exactly what happens:"]
        if random.random() < 0.2:
            phrase = random.choice(precision_phrases)
            response = f"{phrase} {response.lower()}"
        return response

    def _add_educational_depth(self, response: str) -> str:
        """Add educational context"""
        educational_starters = ["Here's some context:", "It's worth noting that", "An interesting aspect is that"]
        if len(response.split()) > 30 and random.random() < 0.3:
            starter = random.choice(educational_starters)
            # Add educational context at the end
            response = f"{response}\n\n{starter} this is a common area where many people have questions."
        return response


# Global instance
conversation_enhancer = ConversationEnhancer()