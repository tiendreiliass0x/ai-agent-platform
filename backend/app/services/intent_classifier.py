"""
Intent Classifier - Determines if query needs RAG or Agentic workflow

Analyzes user messages to route to appropriate system:
- RAG: Information retrieval, Q&A, explanations
- AGENTIC: Actions requiring tool execution (CRM, email, etc.)
"""

import json
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class QueryIntent(str, Enum):
    """Query intent classification"""
    RAG = "rag"  # Information retrieval
    AGENTIC = "agentic"  # Action execution requiring tools


@dataclass
class IntentClassification:
    """Result of intent classification"""
    intent: QueryIntent
    confidence: float  # 0.0 - 1.0
    reasoning: str
    detected_actions: list[str]  # e.g., ["create_lead", "send_email"]
    detected_entities: list[str]  # e.g., ["Salesforce", "john.doe@acme.com"]


class IntentClassifier:
    """LLM-based intent classifier for routing queries"""

    # Action verbs that typically require tool execution
    ACTION_VERBS = {
        "create", "make", "add", "insert", "register", "setup", "initialize",
        "send", "email", "mail", "notify", "message",
        "update", "modify", "change", "edit", "set", "configure",
        "delete", "remove", "cancel", "drop",
        "schedule", "book", "reserve", "plan",
        "approve", "reject", "escalate", "assign",
        "refund", "charge", "pay", "invoice",
        "export", "import", "sync", "backup"
    }

    # External systems that require tool integration
    EXTERNAL_SYSTEMS = {
        "salesforce", "crm", "hubspot", "zendesk",
        "email", "sendgrid", "mailgun", "ses",
        "slack", "teams", "discord",
        "stripe", "paypal", "payment",
        "calendar", "google calendar", "outlook",
        "jira", "asana", "trello",
        "database", "sql", "postgres"
    }

    # Multi-step indicators
    MULTI_STEP_INDICATORS = {
        "and then", "after that", "next", "followed by",
        "then send", "then create", "then update",
        "also", "additionally"
    }

    def __init__(self, llm_service=None):
        """
        Initialize intent classifier.

        Args:
            llm_service: LLM service for semantic analysis (optional)
                        If None, uses heuristic-based classification
        """
        self.llm_service = llm_service

    async def classify(self, message: str, context: Optional[dict] = None) -> IntentClassification:
        """
        Classify user message intent.

        Args:
            message: User message to classify
            context: Optional conversation context

        Returns:
            IntentClassification with routing decision
        """
        # If LLM service available, use semantic analysis
        if self.llm_service:
            return await self._classify_with_llm(message, context)
        else:
            # Fallback to heuristic-based classification
            return self._classify_heuristic(message)

    async def _classify_with_llm(self, message: str, context: Optional[dict]) -> IntentClassification:
        """LLM-based semantic intent classification"""

        prompt = f"""Analyze this user message and determine if it requires TOOL EXECUTION or just INFORMATION RETRIEVAL.

User Message: "{message}"

Classification Guidelines:
1. RAG (Information Retrieval):
   - Questions about products, policies, pricing
   - Explanations or clarifications
   - General knowledge queries
   - "What is...", "How does...", "Tell me about..."

2. AGENTIC (Tool Execution):
   - Creating/updating records in external systems
   - Sending emails, messages, notifications
   - Scheduling, booking, or reserving
   - Multi-step workflows involving external APIs
   - State changes in CRM, databases, etc.

Respond with JSON only:
{{
    "intent": "rag" or "agentic",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "detected_actions": ["action1", "action2"],
    "detected_entities": ["system1", "system2"]
}}

Examples:
- "What are your pricing plans?" → {{"intent": "rag", "confidence": 0.95, "reasoning": "Information query about pricing", "detected_actions": [], "detected_entities": []}}
- "Create a Salesforce lead for John Doe" → {{"intent": "agentic", "confidence": 0.98, "reasoning": "Create action in external CRM system", "detected_actions": ["create_lead"], "detected_entities": ["Salesforce"]}}
- "Send him a welcome email" → {{"intent": "agentic", "confidence": 0.92, "reasoning": "Send action requiring email service", "detected_actions": ["send_email"], "detected_entities": ["email"]}}
"""

        try:
            response = await self.llm_service.generate_response(
                prompt=prompt,
                temperature=0.1,  # Deterministic classification
                max_tokens=200
            )

            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

                return IntentClassification(
                    intent=QueryIntent(result.get("intent", "rag")),
                    confidence=float(result.get("confidence", 0.5)),
                    reasoning=result.get("reasoning", "LLM classification"),
                    detected_actions=result.get("detected_actions", []),
                    detected_entities=result.get("detected_entities", [])
                )
            else:
                # Fallback if JSON parsing fails
                return self._classify_heuristic(message)

        except Exception as e:
            print(f"LLM classification error: {e}")
            # Fallback to heuristic
            return self._classify_heuristic(message)

    def _classify_heuristic(self, message: str) -> IntentClassification:
        """Heuristic-based classification (no LLM required)"""

        message_lower = message.lower()
        detected_actions = []
        detected_entities = []

        # Score based on indicators
        action_score = 0
        info_score = 0

        # Boost for question-style openings
        if re.match(r"^\s*(what|how|why|when|where|who|which|tell me|explain|describe|show me)\b", message_lower):
            info_score += 2

        # Check for action verbs
        for verb in self.ACTION_VERBS:
            if re.search(rf'\b{verb}\b', message_lower):
                action_score += 2
                detected_actions.append(verb)

        # Check for external systems
        for system in self.EXTERNAL_SYSTEMS:
            if system in message_lower:
                action_score += 3
                detected_entities.append(system)

        # Check for multi-step indicators
        for indicator in self.MULTI_STEP_INDICATORS:
            if indicator in message_lower:
                action_score += 2

        # Check for information query patterns
        info_patterns = [
            r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b',
            r'\btell me\b', r'\bexplain\b', r'\bdescribe\b',
            r'\bshow me\b', r'\blist\b', r'\bfind\b',
            r'\b\?\s*$'  # Ends with question mark
        ]

        for pattern in info_patterns:
            if re.search(pattern, message_lower):
                info_score += 1

        # Determine intent based on scores
        if action_score > info_score:
            intent = QueryIntent.AGENTIC
            confidence = min(0.6 + (action_score * 0.1), 0.95)
            reasoning = f"Detected {len(detected_actions)} action verbs and {len(detected_entities)} external systems"
        else:
            intent = QueryIntent.RAG
            confidence = min(0.6 + (info_score * 0.1), 0.95)
            reasoning = "Information retrieval query pattern detected"

        return IntentClassification(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            detected_actions=detected_actions,
            detected_entities=detected_entities
        )

    def is_agentic(self, message: str, threshold: float = 0.7) -> bool:
        """
        Quick synchronous check if message likely needs agentic workflow.

        Args:
            message: User message
            threshold: Confidence threshold for agentic classification

        Returns:
            True if message likely needs tool execution
        """
        classification = self._classify_heuristic(message)
        return classification.intent == QueryIntent.AGENTIC and classification.confidence >= threshold


# Global instance with heuristic-only (no LLM dependency)
intent_classifier = IntentClassifier()
