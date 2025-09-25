"""
Advanced Agent Creation Service
Intelligent agent creation with templates, optimization, and validation.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from app.services.gemini_service import gemini_service
from app.services.database_service import db_service


class AgentType(str, Enum):
    """Predefined agent types with optimized configurations"""
    CUSTOMER_SUPPORT = "customer_support"
    SALES_ASSISTANT = "sales_assistant"
    TECHNICAL_DOCS = "technical_docs"
    LEAD_QUALIFICATION = "lead_qualification"
    PRODUCT_EXPERT = "product_expert"
    FAQ_BOT = "faq_bot"
    ONBOARDING_GUIDE = "onboarding_guide"
    CUSTOM = "custom"


class IndustryType(str, Enum):
    """Industry-specific optimizations"""
    SAAS = "saas"
    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    EDUCATION = "education"
    REAL_ESTATE = "real_estate"
    CONSULTING = "consulting"
    GENERAL = "general"


class AgentCreationService:
    """Advanced agent creation with intelligence and optimization"""

    def __init__(self):
        self.templates = self._load_agent_templates()
        self.industry_optimizations = self._load_industry_optimizations()

    async def create_intelligent_agent(
        self,
        user_id: int,
        organization_id: int,
        agent_data: Dict[str, Any],
        agent_type: AgentType = AgentType.CUSTOM,
        industry: IndustryType = IndustryType.GENERAL,
        auto_optimize: bool = True
    ) -> Dict[str, Any]:
        """Create an agent with intelligent optimization"""

        # 1. Apply template if specified
        if agent_type != AgentType.CUSTOM:
            agent_data = self._apply_agent_template(agent_data, agent_type, industry)

        # 2. Optimize system prompt using AI
        if auto_optimize:
            agent_data = await self._optimize_system_prompt(agent_data, industry)

        # 3. Validate and enhance configuration
        agent_data = self._optimize_agent_config(agent_data, agent_type, industry)

        # 4. Create agent with enhanced metadata
        agent = await db_service.create_agent(
            user_id=user_id,
            organization_id=organization_id,
            name=agent_data["name"],
            description=agent_data["description"],
            system_prompt=agent_data["system_prompt"],
            config=agent_data["config"],
            widget_config=agent_data["widget_config"]
        )

        # 5. Generate embedding code and setup instructions
        embed_code = self._generate_embed_code(agent)
        setup_guide = self._generate_setup_guide(agent, agent_type, industry)

        return {
            "agent": agent,
            "embed_code": embed_code,
            "setup_guide": setup_guide,
            "optimization_applied": auto_optimize,
            "template_used": agent_type.value if agent_type != AgentType.CUSTOM else None,
            "recommendations": self._generate_recommendations(agent_data, agent_type, industry)
        }

    def _apply_agent_template(
        self,
        agent_data: Dict[str, Any],
        agent_type: AgentType,
        industry: IndustryType
    ) -> Dict[str, Any]:
        """Apply predefined template for agent type"""

        template = self.templates.get(agent_type, {})
        industry_opt = self.industry_optimizations.get(industry, {})

        # Merge template with user data (user data takes precedence)
        enhanced_data = template.copy()
        enhanced_data.update(agent_data)

        # Apply industry-specific optimizations
        if industry_opt:
            enhanced_data["config"].update(industry_opt.get("config", {}))
            enhanced_data["widget_config"].update(industry_opt.get("widget_config", {}))

            # Enhance system prompt with industry context
            if industry_opt.get("system_prompt_enhancement"):
                enhanced_data["system_prompt"] = (
                    enhanced_data.get("system_prompt", "") +
                    "\n\n" +
                    industry_opt["system_prompt_enhancement"]
                )

        return enhanced_data

    async def _optimize_system_prompt(
        self,
        agent_data: Dict[str, Any],
        industry: IndustryType
    ) -> Dict[str, Any]:
        """Use AI to optimize the system prompt"""

        try:
            optimization_prompt = f"""
You are an expert AI prompt engineer. Optimize this agent system prompt for maximum effectiveness.

Current Details:
- Agent Name: {agent_data.get('name', 'AI Assistant')}
- Description: {agent_data.get('description', 'No description provided')}
- Industry: {industry.value}
- Current System Prompt: {agent_data.get('system_prompt', 'You are a helpful assistant.')}

Please optimize the system prompt following these guidelines:
1. Be specific about the agent's role and expertise
2. Include industry-specific knowledge and terminology
3. Set clear behavioral expectations
4. Add guardrails for appropriate responses
5. Make it concise but comprehensive (max 500 words)

Return only the optimized system prompt, no explanations.
"""

            optimized_prompt = await gemini_service.generate_response(
                prompt=optimization_prompt,
                temperature=0.3,
                max_tokens=800
            )

            agent_data["system_prompt"] = optimized_prompt.strip()
            agent_data["_optimization_applied"] = True

        except Exception as e:
            print(f"Failed to optimize system prompt: {e}")
            # Continue with original prompt if optimization fails

        return agent_data

    def _optimize_agent_config(
        self,
        agent_data: Dict[str, Any],
        agent_type: AgentType,
        industry: IndustryType
    ) -> Dict[str, Any]:
        """Optimize agent configuration based on type and industry"""

        # Default config
        config = {
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.7,
            "max_tokens": 1000,
            "memory_enabled": True,
            "context_optimization": True,
            "response_time_target": 2.0,  # seconds
            "quality_threshold": 0.8
        }

        # Agent-type specific optimizations
        type_configs = {
            AgentType.CUSTOMER_SUPPORT: {
                "temperature": 0.5,  # More consistent responses
                "max_tokens": 800,
                "escalation_enabled": True,
                "sentiment_analysis": True
            },
            AgentType.SALES_ASSISTANT: {
                "temperature": 0.6,
                "persuasion_mode": True,
                "lead_scoring": True,
                "follow_up_enabled": True
            },
            AgentType.TECHNICAL_DOCS: {
                "temperature": 0.3,  # More precise
                "max_tokens": 1500,
                "code_highlighting": True,
                "technical_accuracy": "high"
            },
            AgentType.FAQ_BOT: {
                "temperature": 0.2,  # Very consistent
                "max_tokens": 500,
                "exact_match_priority": True
            }
        }

        # Industry-specific optimizations
        industry_configs = {
            IndustryType.FINTECH: {
                "compliance_mode": True,
                "security_level": "high",
                "regulatory_awareness": True
            },
            IndustryType.HEALTHCARE: {
                "privacy_mode": "strict",
                "medical_disclaimer": True,
                "hipaa_compliant": True
            },
            IndustryType.ECOMMERCE: {
                "product_recommendations": True,
                "inventory_awareness": True,
                "checkout_assistance": True
            }
        }

        # Apply optimizations
        config.update(agent_data.get("config", {}))
        config.update(type_configs.get(agent_type, {}))
        config.update(industry_configs.get(industry, {}))

        agent_data["config"] = config

        # Optimize widget config
        widget_config = {
            "theme": "modern",
            "position": "bottom-right",
            "size": "medium",
            "animation": "slide-up",
            "branding": True,
            "sound_enabled": False,
            "typing_indicator": True,
            "quick_replies": True
        }

        widget_config.update(agent_data.get("widget_config", {}))
        agent_data["widget_config"] = widget_config

        return agent_data

    def _generate_embed_code(self, agent) -> str:
        """Generate JavaScript embed code for the agent"""

        return f"""<!-- AI Agent Embed Code -->
<script>
  (function() {{
    var script = document.createElement('script');
    script.src = 'https://cdn.yourdomain.com/agent-widget.js';
    script.async = true;
    script.onload = function() {{
      YourAgent.init({{
        agentId: '{agent.id}',
        apiKey: '{agent.api_key}',
        config: {agent.widget_config}
      }});
    }};
    document.head.appendChild(script);
  }})();
</script>
<!-- End AI Agent Embed Code -->"""

    def _generate_setup_guide(
        self,
        agent,
        agent_type: AgentType,
        industry: IndustryType
    ) -> Dict[str, Any]:
        """Generate comprehensive setup guide"""

        return {
            "steps": [
                {
                    "title": "Add Embed Code",
                    "description": "Copy the embed code to your website's HTML",
                    "code": self._generate_embed_code(agent),
                    "estimated_time": "2 minutes"
                },
                {
                    "title": "Upload Knowledge Base",
                    "description": "Add relevant documents and URLs for your agent to learn from",
                    "action": "upload_documents",
                    "estimated_time": "10-30 minutes"
                },
                {
                    "title": "Test & Optimize",
                    "description": "Chat with your agent and refine responses",
                    "action": "test_agent",
                    "estimated_time": "15 minutes"
                },
                {
                    "title": "Monitor Performance",
                    "description": "Review analytics and customer feedback",
                    "action": "view_analytics",
                    "estimated_time": "Ongoing"
                }
            ],
            "best_practices": self._get_best_practices(agent_type, industry),
            "common_issues": self._get_common_issues(agent_type),
            "optimization_tips": self._get_optimization_tips(agent_type, industry)
        }

    def _generate_recommendations(
        self,
        agent_data: Dict[str, Any],
        agent_type: AgentType,
        industry: IndustryType
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations"""

        recommendations = []

        # Document upload recommendations
        if agent_type == AgentType.CUSTOMER_SUPPORT:
            recommendations.append({
                "type": "document_upload",
                "priority": "high",
                "title": "Upload FAQ and Support Documentation",
                "description": "Add your most common support questions and answers to improve response quality."
            })

        if industry == IndustryType.FINTECH:
            recommendations.append({
                "type": "compliance",
                "priority": "high",
                "title": "Review Compliance Settings",
                "description": "Ensure your agent complies with financial regulations in your jurisdiction."
            })

        # Performance recommendations
        recommendations.append({
            "type": "testing",
            "priority": "medium",
            "title": "Test Common Scenarios",
            "description": "Run through typical customer interactions to validate agent responses."
        })

        return recommendations

    def _load_agent_templates(self) -> Dict[AgentType, Dict[str, Any]]:
        """Load predefined agent templates"""

        return {
            AgentType.CUSTOMER_SUPPORT: {
                "description": "Helps customers with questions, issues, and support requests",
                "system_prompt": """You are a professional customer support agent. Your role is to:

1. Help customers resolve issues quickly and effectively
2. Provide accurate information about products and services
3. Escalate complex issues to human agents when necessary
4. Maintain a helpful, patient, and professional tone
5. Follow company policies and procedures

Always be empathetic and solution-focused. If you cannot resolve an issue, clearly explain next steps and how to reach human support.""",
                "config": {
                    "temperature": 0.5,
                    "max_tokens": 800,
                    "escalation_enabled": True
                },
                "widget_config": {
                    "welcome_message": "Hi! I'm here to help with any questions or issues you might have.",
                    "theme": "professional",
                    "quick_replies": ["Get Help", "Track Order", "Contact Support"]
                }
            },

            AgentType.SALES_ASSISTANT: {
                "description": "Qualifies leads and assists with sales inquiries",
                "system_prompt": """You are a sales assistant focused on helping potential customers. Your goals are to:

1. Understand customer needs and pain points
2. Present relevant solutions and benefits
3. Qualify leads and gather contact information
4. Schedule demos or consultations when appropriate
5. Build trust and rapport with prospects

Be consultative rather than pushy. Focus on value and how you can solve customer problems.""",
                "config": {
                    "temperature": 0.6,
                    "lead_scoring": True,
                    "follow_up_enabled": True
                },
                "widget_config": {
                    "welcome_message": "Hello! I'd love to learn about your needs and see how we can help.",
                    "theme": "modern",
                    "quick_replies": ["Learn More", "Get Pricing", "Schedule Demo"]
                }
            },

            AgentType.TECHNICAL_DOCS: {
                "description": "Provides technical documentation and developer support",
                "system_prompt": """You are a technical documentation assistant for developers. Your responsibilities include:

1. Explaining technical concepts clearly and accurately
2. Providing code examples and implementation guidance
3. Helping with API documentation and integration
4. Troubleshooting technical issues
5. Directing users to relevant documentation sections

Be precise and technical when needed, but also explain complex concepts in accessible terms.""",
                "config": {
                    "temperature": 0.3,
                    "max_tokens": 1500,
                    "code_highlighting": True
                },
                "widget_config": {
                    "welcome_message": "I'm here to help with technical questions and documentation.",
                    "theme": "developer",
                    "quick_replies": ["API Docs", "Code Examples", "Troubleshooting"]
                }
            }
        }

    def _load_industry_optimizations(self) -> Dict[IndustryType, Dict[str, Any]]:
        """Load industry-specific optimizations"""

        return {
            IndustryType.FINTECH: {
                "system_prompt_enhancement": """

IMPORTANT: You are operating in the financial services industry. Always:
- Emphasize security and data protection
- Include appropriate disclaimers for financial advice
- Be aware of regulatory compliance requirements
- Never provide specific investment advice
- Direct users to licensed professionals for complex financial matters""",
                "config": {
                    "compliance_mode": True,
                    "security_level": "high"
                },
                "widget_config": {
                    "disclaimer_enabled": True,
                    "security_badge": True
                }
            },

            IndustryType.HEALTHCARE: {
                "system_prompt_enhancement": """

IMPORTANT: You are operating in healthcare. Always:
- Include medical disclaimers - you cannot provide medical advice
- Respect patient privacy and confidentiality
- Direct users to healthcare professionals for medical questions
- Be sensitive to health-related concerns
- Follow HIPAA guidelines for any patient information""",
                "config": {
                    "medical_disclaimer": True,
                    "privacy_mode": "strict"
                }
            },

            IndustryType.ECOMMERCE: {
                "config": {
                    "product_recommendations": True,
                    "inventory_awareness": True
                },
                "widget_config": {
                    "shopping_assistant": True,
                    "quick_replies": ["Track Order", "Returns", "Product Info"]
                }
            }
        }

    def _get_best_practices(self, agent_type: AgentType, industry: IndustryType) -> List[str]:
        """Get best practices for agent type and industry"""

        practices = [
            "Upload comprehensive and up-to-date documentation",
            "Test your agent regularly with real customer scenarios",
            "Monitor conversation quality and user satisfaction",
            "Keep your knowledge base current and accurate",
            "Review and refine your system prompt based on feedback"
        ]

        if agent_type == AgentType.CUSTOMER_SUPPORT:
            practices.extend([
                "Set up clear escalation paths to human agents",
                "Include common troubleshooting steps in your knowledge base",
                "Train your agent on your return and refund policies"
            ])

        if industry == IndustryType.FINTECH:
            practices.extend([
                "Ensure all compliance disclaimers are current",
                "Regular security and privacy reviews",
                "Keep regulatory information up to date"
            ])

        return practices

    def _get_common_issues(self, agent_type: AgentType) -> List[Dict[str, str]]:
        """Get common issues and solutions"""

        return [
            {
                "issue": "Agent gives incorrect information",
                "solution": "Review and update your knowledge base with accurate information"
            },
            {
                "issue": "Agent doesn't understand customer questions",
                "solution": "Add more examples and variations of common questions to your training data"
            },
            {
                "issue": "Agent is too generic",
                "solution": "Refine your system prompt to be more specific about your business and tone"
            }
        ]

    def _get_optimization_tips(self, agent_type: AgentType, industry: IndustryType) -> List[str]:
        """Get optimization tips"""

        return [
            "Use specific examples in your system prompt",
            "Regularly review conversation logs for improvement opportunities",
            "A/B test different system prompts to find what works best",
            "Keep responses concise but complete",
            "Use your brand voice and personality consistently"
        ]


# Global instance
agent_creation_service = AgentCreationService()