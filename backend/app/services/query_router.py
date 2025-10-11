"""
Query Router - Routes queries to RAG or Agentic workflow

Intelligent dispatcher that analyzes user intent and routes to:
- RAG Service: Information retrieval (Phase 1+2)
- Agent Orchestrator: Tool execution (Phase 3)
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from .intent_classifier import IntentClassifier, QueryIntent
from .rag_service import RAGService
from app.orchestrator.models import AgentTask, AgentContext, AgentUser, OrchestrationResult
from app.models.agent import Agent
from app.tooling.bootstrap import ensure_default_tools_registered
from app.executor import ToolExecutionError


@dataclass
class RoutedResponse:
    """Unified response from either RAG or Agentic workflow"""
    response: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    routing_decision: str  # "rag" or "agentic"
    reasoning: str
    tools_used: List[str]
    execution_time_ms: float
    metadata: Dict[str, Any]


class QueryRouter:
    """Intelligent router for RAG vs Agentic workflows"""

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        orchestrator = None,
        rag_service: Optional[RAGService] = None,
        enable_agentic: bool = True,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize query router.

        Args:
            intent_classifier: Intent classification service
            orchestrator: Agent orchestrator for tool execution
            rag_service: RAG service for information retrieval
            enable_agentic: Enable agentic workflow routing (default: True)
            confidence_threshold: Minimum confidence for agentic routing (default: 0.7)
        """
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.orchestrator = orchestrator
        self.rag_service = rag_service or RAGService()
        self.enable_agentic = enable_agentic
        self.confidence_threshold = confidence_threshold

    async def route(
        self,
        message: str,
        agent: Agent,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str,
        agent_config: Dict[str, Any],
        db_session: AsyncSession,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutedResponse:
        """
        Route query to appropriate system (RAG or Agentic).

        Args:
            message: User message
            agent: Agent configuration
            conversation_history: Conversation context
            system_prompt: System prompt for LLM
            agent_config: Agent configuration
            db_session: Database session
            context: Additional context

        Returns:
            RoutedResponse with unified format
        """
        import time
        start_time = time.time()

        # Classify intent
        classification = await self.intent_classifier.classify(message, context)

        # Route based on intent and configuration
        if (
            self.enable_agentic
            and classification.intent == QueryIntent.AGENTIC
            and classification.confidence >= self.confidence_threshold
            and self.orchestrator is not None
        ):
            # Route to Agentic workflow
            result = await self._route_to_agentic(
                message=message,
                agent=agent,
                context=context or {},
                classification=classification
            )
        else:
            # Route to RAG (default)
            result = await self._route_to_rag(
                message=message,
                agent_id=agent.id,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
                agent_config=agent_config,
                db_session=db_session,
                classification=classification
            )

        execution_time_ms = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time_ms

        return result

    async def _route_to_agentic(
        self,
        message: str,
        agent: Agent,
        context: Dict[str, Any],
        classification
    ) -> RoutedResponse:
        """Route to Agent Orchestrator for tool execution"""

        # Create agent task from user message
        task = AgentTask(
            description=message,
            metadata={
                "agent_id": agent.id,
                "detected_actions": classification.detected_actions,
                "detected_entities": classification.detected_entities,
                **context
            }
        )

        # Create agent context with user permissions
        agent_context = AgentContext(
            user=AgentUser(
                id=agent.id,
                name=agent.name,
                permissions=self._get_agent_permissions(agent)
            ),
            session_id=context.get("session_id"),
            conversation_id=context.get("conversation_id"),
            metadata=context
        )

        # Execute via orchestrator
        try:
            if hasattr(self.orchestrator, "tool_registry"):
                await ensure_default_tools_registered(self.orchestrator.tool_registry)

            orchestration_result: OrchestrationResult = await self.orchestrator.run_task(
                context=agent_context,
                task=task
            )
        except RuntimeError as exc:
            return self._agentic_failure_response(
                error_message=str(exc),
                classification=classification,
                status="error_missing_credentials",
            )
        except ToolExecutionError as exc:
            return self._agentic_failure_response(
                error_message=str(exc),
                classification=classification,
                status="error_tool_execution",
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._agentic_failure_response(
                error_message=str(exc),
                classification=classification,
                status="error_unexpected",
            )

        # Format orchestration result into unified response
        return self._format_orchestration_result(
            orchestration_result,
            classification
        )

    async def _route_to_rag(
        self,
        message: str,
        agent_id: int,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str,
        agent_config: Dict[str, Any],
        db_session: AsyncSession,
        classification
    ) -> RoutedResponse:
        """Route to RAG service for information retrieval"""

        # Generate response via RAG
        rag_response = await self.rag_service.generate_response(
            query=message,
            agent_id=agent_id,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            agent_config=agent_config,
            db_session=db_session
        )

        return RoutedResponse(
            response=rag_response["response"],
            confidence_score=rag_response.get("confidence_score", 0.8),
            sources=rag_response.get("sources", []),
            routing_decision="rag",
            reasoning=classification.reasoning,
            tools_used=[],
            execution_time_ms=0.0,  # Will be set by caller
            metadata={
                "personality_applied": rag_response.get("personality_applied", False),
                "context_used": rag_response.get("context_used", False),
                "classification": {
                    "intent": classification.intent.value,
                    "confidence": classification.confidence
                }
            }
        )

    def _format_orchestration_result(
        self,
        result: OrchestrationResult,
        classification
    ) -> RoutedResponse:
        """Format orchestration result into unified response"""

        if result.status == "completed":
            # Extract response from step results
            response_parts = []
            tools_used = []
            sources = []

            plan_steps = {step.id: step for step in result.plan.steps}
            for step_id, step_result in result.step_results.items():
                if step_result.get("status") != "completed":
                    continue

                plan_step = plan_steps.get(step_id)
                action = getattr(plan_step, "action", "") if plan_step else ""
                if action and "." in action:
                    tools_used.append(action.split(".", 1)[0])

                step_data = step_result.get("result", {})
                summary = f"✓ Completed {step_id}"
                if isinstance(step_data, dict) and step_data:
                    details = ", ".join(f"{k}: {v}" for k, v in step_data.items())
                    summary = f"{summary} ({details})"
                response_parts.append(summary)

                if isinstance(step_data, dict):
                    for key, value in step_data.items():
                        if key in {"id", "ticket_id", "message_id", "lead_id"}:
                            sources.append({
                                "type": "tool_execution",
                                "step": step_id,
                                "data": {key: value}
                            })

            response_text = "\n".join(response_parts) if response_parts else "Task completed successfully"

            return RoutedResponse(
                response=response_text,
                confidence_score=0.95,  # High confidence for successful execution
                sources=sources,
                routing_decision="agentic",
                reasoning=classification.reasoning,
                tools_used=list(set(tools_used)),
                execution_time_ms=0.0,  # Will be set by caller
                metadata={
                    "plan_goal": result.plan.goal,
                    "strategy": result.plan.strategy,
                    "steps_completed": len([s for s in result.step_results.values() if s.get("status") == "completed"]),
                    "total_steps": len(result.step_results),
                    "classification": {
                        "intent": classification.intent.value,
                        "confidence": classification.confidence
                    }
                }
            )

        else:
            # Handle failure
            error_message = result.error or "Execution failed"

            return self._agentic_failure_response(
                error_message=error_message,
                classification=classification,
                status=result.status,
            )

    def _get_agent_permissions(self, agent: Agent) -> set:
        """
        Extract permissions from agent configuration.

        Args:
            agent: Agent model

        Returns:
            Set of permission strings for RBAC
        """
        # Extract from agent config if available
        config = agent.config or {}
        custom_permissions = config.get("permissions", [])

        if custom_permissions:
            return set(custom_permissions)

        # Based on agent tier
        if hasattr(agent, 'tier'):
            if agent.tier == "enterprise":
                return {"*"}  # Full access
            elif agent.tier == "professional":
                return {
                    "crm.read", "crm.create_ticket", "crm.update_ticket",
                    "email.send_transactional",
                    "calendar.read", "calendar.create_event"
                }
            else:  # basic
                return {"crm.read", "email.send_transactional"}

        return set()

    def _agentic_failure_response(
        self,
        error_message: str,
        classification,
        status: str,
    ) -> RoutedResponse:
        return RoutedResponse(
            response=f"❌ Unable to complete the requested action: {error_message}",
            confidence_score=0.3,
            sources=[],
            routing_decision="agentic",
            reasoning=classification.reasoning,
            tools_used=[],
            execution_time_ms=0.0,
            metadata={
                "error": error_message,
                "status": status,
                "classification": {
                    "intent": classification.intent.value,
                    "confidence": classification.confidence,
                },
            },
        )

    async def should_use_agentic(self, message: str) -> bool:
        """
        Quick check if message should use agentic workflow.

        Args:
            message: User message

        Returns:
            True if agentic workflow recommended
        """
        if not self.enable_agentic or self.orchestrator is None:
            return False

        classification = await self.intent_classifier.classify(message)
        return (
            classification.intent == QueryIntent.AGENTIC
            and classification.confidence >= self.confidence_threshold
        )


# Global router instance (will be initialized with dependencies)
_router_instance: Optional[QueryRouter] = None


def get_query_router(
    orchestrator=None,
    enable_agentic: bool = True
) -> QueryRouter:
    """
    Get or create global query router instance.

    Args:
        orchestrator: Agent orchestrator (optional, will use default if None)
        enable_agentic: Enable agentic routing

    Returns:
        QueryRouter instance
    """
    global _router_instance

    if _router_instance is None:
        # Import here to avoid circular dependency
        from .intent_classifier import IntentClassifier
        from app.services.gemini_service import gemini_service
        from app.services.orchestrator_builder import get_agent_orchestrator
        from app.services.rag_service import RAGService

        if enable_agentic and orchestrator is None:
            orchestrator = get_agent_orchestrator()

        # Create classifier with LLM support
        classifier = IntentClassifier(llm_service=gemini_service)

        # Create router
        _router_instance = QueryRouter(
            intent_classifier=classifier,
            orchestrator=orchestrator,  # May be None if agentic disabled
            rag_service=RAGService(),
            enable_agentic=enable_agentic
        )

    return _router_instance
