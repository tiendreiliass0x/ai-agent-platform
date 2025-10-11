"""Agentic Support Response Demo

Run a miniature end-to-end flow where the query router classifies an action
request, invokes the agent orchestrator (reasoning + execution), and uses a
tool adapter to simulate CRM operations.

Run:
    uv run python -m demo.agentic_support_response_demo
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List

from app.orchestrator import AgentContext, AgentTask, AgentUser
from app.orchestrator.agent_orchestrator import AgentOrchestrator
from app.orchestrator.learning import LearningSystem
from app.orchestrator.memory_provider import EphemeralMemoryProvider
from app.orchestrator.strategy import StrategySelector
from app.orchestrator.task_tracker import TaskTracker
from app.planner.models import Plan, Step
from app.policy.engine import GateResult
from app.reasoning import ReasoningEngine
from app.reasoning.engine import BaseReasoner
from app.services.intent_classifier import IntentClassifier, IntentClassification, QueryIntent
from app.services.query_router import QueryRouter, RoutedResponse
from app.tooling.adapters import adapter_registry, ToolAdapter, ToolExecutionContext
from app.tooling.adapters.registry import AdapterRegistry
from app.tooling.models import OperationSpec, SideEffect, ToolManifest
from app.verifier import Verifier


# --- Demo Adapter -----------------------------------------------------------------


class DemoCRMAdapter(ToolAdapter):
    """Simulates CRM operations for the demo."""

    async def execute(
        self,
        manifest: ToolManifest,
        operation: OperationSpec,
        args: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> Dict[str, Any]:
        if operation.op_id == "crm.get_ticket":
            return {
                "ticket_id": args.get("ticket_id"),
                "customer_id": "CUST-9987",
                "subject": "Shipment still not here",
                "details": "Promised delivery by Friday; still waiting.",
            }
        if operation.op_id == "crm.get_customer":
            return {
                "id": args.get("customer_id"),
                "name": "Maya Carter",
                "tenure_years": 3,
                "issue_history": [],
            }
        if operation.op_id == "crm.suggest_resolution":
            return {
                "options": [
                    "Full refund plus expedited replacement.",
                    "30% discount on next order with apology.",
                ]
            }
        if operation.op_id == "crm.draft_response":
            return {
                "draft": (
                    "Hi Maya,\n\n"
                    "I’m so sorry the replacement is still en route. I've processed a full refund "
                    "and placed an expedited shipment that will land within two days. "
                    "I'll watch the tracking and follow up tomorrow.\n\n"
                    "Thank you for your patience,\nSarah"
                )
            }
        return {"message": f"Unhandled operation {operation.op_id}"}


def register_demo_adapter(registry: AdapterRegistry) -> None:
    def factory(metadata: Dict[str, Any]):
        adapter_name = metadata.get("integrations", {}).get("crm", {}).get("adapter")
        if adapter_name == "demo_crm":
            return DemoCRMAdapter()
        return None

    registry.register("crm", factory)


# --- Stubs & Helpers ----------------------------------------------------------------


class DemoToolRegistry:
    def __init__(self) -> None:
        self.manifest = _build_crm_manifest()

    async def get_tool(self, name: str, version: str | None = None) -> ToolManifest:
        return self.manifest

    async def list_tools(self):
        return [self.manifest]

    async def register_manifest_from_yaml(self, manifest_yaml: str, source: str | None = None):
        return self.manifest


class DemoStrategySelector(StrategySelector):
    async def select(self, task: AgentTask, context: AgentContext):
        return type("Decision", (), {"name": "plan_execute", "reasoning": "demo"})()


class DemoContextBuilder:
    async def build(self, *args, **kwargs):
        return "Demo context"


class DemoPolicyEngine:
    async def gate(self, user, operation, args, manifest):
        return GateResult(decision="allow", reasoning="demo", risk_score=0.0)


class DemoReasoner(BaseReasoner):
    async def generate_plan(self, task: AgentTask, **_: Any) -> Plan:
        text = task.description.lower()
        if "ticket" in text:
            return Plan(
                goal="Gather context and propose resolution options",
                strategy="plan_execute",
                steps=[
                    Step(id="read_ticket", kind="tool", action="crm.get_ticket", args={"ticket_id": "4521"}),
                    Step(
                        id="customer_profile",
                        kind="tool",
                        action="crm.get_customer",
                        args={"customer_id": "CUST-9987"},
                        depends_on=["read_ticket"],
                    ),
                    Step(
                        id="suggest_options",
                        kind="tool",
                        action="crm.suggest_resolution",
                        args={"issue": "delayed shipment"},
                        depends_on=["customer_profile"],
                    ),
                ],
            )
        return Plan(
            goal="Draft warm response",
            strategy="plan_execute",
            steps=[
                Step(
                    id="draft_email",
                    kind="tool",
                    action="crm.draft_response",
                    args={"ticket_id": "4521", "tone": "warm", "resolution": "Full refund"},
                )
            ],
        )


def _build_crm_manifest() -> ToolManifest:
    operations = {
        "crm.get_ticket": OperationSpec(
            op_id="crm.get_ticket",
            method="GET",
            path="/v1/tickets/{ticket_id}",
            side_effect=SideEffect.READ,
            description="Retrieve ticket details",
            args_schema={},
            returns={},
        ),
        "crm.get_customer": OperationSpec(
            op_id="crm.get_customer",
            method="GET",
            path="/v1/customers/{customer_id}",
            side_effect=SideEffect.READ,
            description="Fetch customer profile",
            args_schema={},
            returns={},
        ),
        "crm.suggest_resolution": OperationSpec(
            op_id="crm.suggest_resolution",
            method="POST",
            path="/v1/resolution_suggestions",
            side_effect=SideEffect.READ,
            description="Suggest resolution options",
            args_schema={},
            returns={},
        ),
        "crm.draft_response": OperationSpec(
            op_id="crm.draft_response",
            method="POST",
            path="/v1/tickets/{ticket_id}/draft",
            side_effect=SideEffect.WRITE,
            description="Draft a response email",
            args_schema={},
            returns={},
        ),
    }
    return ToolManifest(
        name="crm",
        version="demo",
        display_name="Demo CRM",
        description="In-memory CRM for demo",
        auth={},
        operations=operations,
        schemas={},
        rate_limits={},
        governance={},
    )


# --- Demo execution ------------------------------------------------------------------


async def build_demo_orchestrator() -> AgentOrchestrator:
    register_demo_adapter(adapter_registry)

    reasoning_engine = ReasoningEngine({"plan_execute": DemoReasoner()})

    executor = DemoExecutor()
    orchestrator = AgentOrchestrator(
        reasoning_engine=reasoning_engine,
        executor=executor,
        tool_registry=DemoToolRegistry(),
        strategy_selector=DemoStrategySelector(),
        context_builder=DemoContextBuilder(),
        memory_provider=EphemeralMemoryProvider(),
        learning_system=LearningSystem(),
        policy_engine=DemoPolicyEngine(),
    )
    orchestrator.executor.adapter_registry = adapter_registry
    return orchestrator


class DemoExecutor:
    def __init__(self):
        self.stub_executor = ExecutorStub()
        self.adapter_registry = adapter_registry

    async def execute(self, step, user, context):
        return await self.stub_executor.execute(step, user, context)


class ExecutorStub:
    async def execute(self, step, user, context):
        adapter = adapter_registry.get_adapter(step.action.split(".")[0], context.get("_agent_metadata", {}))
        if adapter:
            manifest = _build_crm_manifest()
            operation = manifest.operations[step.action]
            execution_context = ToolExecutionContext(
                user=user,
                shared_state=context,
                agent_metadata=context.get("_agent_metadata", {}),
                secret_store=None,
                http_client=None,
            )
            return await adapter.execute(manifest, operation, step.args, execution_context)
        return {"status": "ok"}


async def build_query_router(orchestrator: AgentOrchestrator) -> QueryRouter:
    classifier = DemoClassifier()
    return QueryRouter(
        intent_classifier=classifier,
        orchestrator=orchestrator,
        rag_service=StubRAGService(),
        enable_agentic=True,
        confidence_threshold=0.6,
    )


class DemoClassifier(IntentClassifier):
    async def classify(self, message: str, context: Dict[str, Any] | None = None) -> IntentClassification:
        lower = message.lower()
        if "ticket" in lower or "option" in lower:
            return IntentClassification(
                intent=QueryIntent.AGENTIC,
                confidence=0.92,
                reasoning="Detected support ticket workflow",
                detected_actions=["support_response"],
                detected_entities=["crm"],
            )
        return IntentClassification(
            intent=QueryIntent.RAG,
            confidence=0.5,
            reasoning="Default to RAG",
            detected_actions=[],
            detected_entities=[],
        )


class StubRAGService:
    async def generate_response(self, **_: Any) -> Dict[str, Any]:
        return {"response": "Here is the latest policy information.", "confidence_score": 0.6, "sources": []}


# --- Conversation -------------------------------------------------------------------


async def run_demo() -> None:
    orchestrator = await build_demo_orchestrator()
    router = await build_query_router(orchestrator)

    agent = AgentStub(
        id=1,
        name="CX Agent",
        config={
            "permissions": [
                "crm.get_ticket",
                "crm.get_customer",
                "crm.suggest_resolution",
                "crm.draft_response",
            ]
        },
    )

    conversation: List[Dict[str, str]] = []
    context_metadata = {"session_id": "demo-session", "integrations": {"crm": {"adapter": "demo_crm"}}}

    user_utterances = [
        "I need to respond to the angry customer in ticket #4521",
        "Option 1, but make it warm",
    ]

    for message in user_utterances:
        response: RoutedResponse = await router.route(
            message=message,
            agent=agent,
            conversation_history=conversation,
            system_prompt="You are an empathetic support assistant.",
            agent_config=agent.config,
            db_session=None,
            context=context_metadata,
        )

        print(f"\nUser: {message}")
        print(f"Routing decision: {response.routing_decision}")
        print(f"Agent: {response.response}\n")

        conversation.append({"role": "user", "content": message})
        conversation.append({"role": "assistant", "content": response.response})


@dataclass
class AgentStub:
    id: int
    name: str
    config: Dict[str, Any]
    tier: str = "professional"


if __name__ == "__main__":
    asyncio.run(run_demo())
