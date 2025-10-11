import json
import pytest

from app.context_engine.conversation_memory import ConversationMemory, ConversationTurn
from app.orchestrator import AgentContext, AgentTask, AgentUser
from app.orchestrator.learning import LearningSystem
from app.orchestrator.memory_provider import EphemeralMemoryProvider, PersistentMemoryProvider
from app.planner.models import Plan, Step
from app.services.orchestrator_builder import create_agent_orchestrator
from app.tooling.models import OperationSpec, SideEffect, ToolManifest


class StubPlanner:
    def __init__(self, action_id: str):
        self.action_id = action_id

    async def plan(self, task, strategy, context):
        return Plan(
            goal=f"Execute {self.action_id}",
            strategy=strategy,
            steps=[
                Step(
                    id="step_1",
                    kind="tool",
                    action=self.action_id,
                    args={"value": 1},
                )
            ],
        )


class StubLLMService:
    async def generate_response(self, *args, **kwargs):
        plan = {
            "goal": "React Plan",
            "steps": [
                {"id": "step_r", "kind": "tool", "action": "test.tool", "args": {}}
            ],
        }
        return json.dumps(plan)


class StubToolRegistry:
    def __init__(self, manifest: ToolManifest):
        self.manifest = manifest

    async def search_tools(self, description, top_k=10):
        return [self.manifest]

    async def list_tools(self):
        return [self.manifest]

    async def get_tool(self, name: str, version: str = None):
        return self.manifest


class StubStrategySelector:
    async def select(self, task, context):
        return type("Decision", (), {"name": "plan_execute", "reasoning": "stub"})()


class StubSecretStore:
    async def get_oauth_token(self, user, tool_name: str) -> str:
        return "secret-token"

    async def get_api_key(self, user, tool_name: str) -> str:
        return "secret-api-key"


class StubHTTPClient:
    async def request(self, method, url, headers=None, json=None, timeout=None):
        return {"_duration_ms": 1, "status": "ok"}


class StubRBAC:
    async def has_permission(self, user, op_id: str) -> bool:
        return True

    async def can_access_pii(self, user) -> bool:
        return True


class StubAuditLogger:
    async def log_gate_decision(self, **kwargs) -> None:
        return None


class StubTelemetry:
    async def log_action(self, **kwargs) -> None:
        return None


class StubCircuitBreaker:
    async def call(self, tool_name: str, func, *args, **kwargs):
        return await func(*args, **kwargs)


@pytest.mark.asyncio
async def test_create_agent_orchestrator_with_stubs_executes_plan():
    action_id = "test.tool_action"
    manifest = ToolManifest(
        name="test.tool",
        version="1.0",
        display_name="Test Tool",
        description="Demo",
        auth={},
        operations={
            action_id: OperationSpec(
                op_id=action_id,
                method="POST",
                path="/execute",
                side_effect=SideEffect.WRITE,
                description="Test action",
                args_schema={},
                returns={},
                preconditions=[],
                postconditions=[],
            )
        },
        schemas={},
        rate_limits={},
        governance={},
    )

    orchestrator = create_agent_orchestrator(
        llm_service=StubLLMService(),
        tool_registry_instance=StubToolRegistry(manifest),
        planner=StubPlanner(action_id),
        strategy_selector=StubStrategySelector(),
        memory_provider=EphemeralMemoryProvider(),
        learning_system=LearningSystem(),
        secret_store=StubSecretStore(),
        http_client=StubHTTPClient(),
        rbac_service=StubRBAC(),
        audit_logger=StubAuditLogger(),
        telemetry=StubTelemetry(),
        circuit_breaker=StubCircuitBreaker(),
    )

    context = AgentContext(user=AgentUser(id="user"), session_id="session-1")
    task = AgentTask(description="Run tool")

    result = await orchestrator.run_task(context, task)

    assert result.status == "completed"
    assert result.step_results["step_1"]["status"] == "completed"
    assert result.step_results["step_1"]["result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_learning_system_records_history():
    learner = LearningSystem(max_records=2)
    plan = Plan(goal="Test", strategy="plan_execute", steps=[])

    await learner.record_execution(plan, True)
    await learner.record_execution(plan, False)
    await learner.record_execution(plan, True)

    assert len(learner.history) == 2
    assert list(entry["success"] for entry in learner.history) == [False, True]


@pytest.mark.asyncio
async def test_ephemeral_memory_provider_creates_unique_memory():
    provider = EphemeralMemoryProvider(max_turns=5)
    context_a = AgentContext(user=AgentUser(id=1), session_id="abc")
    context_b = AgentContext(user=AgentUser(id=2), session_id="def")

    memory_a = await provider.load(context_a)
    memory_b = await provider.load(context_b)

    assert memory_a is not memory_b
    assert memory_a.max_turns == 5
    assert memory_b.max_turns == 5

    await provider.save(context_a, memory_a)  # should not raise


class StubConversationStore:
    def __init__(self):
        self.turns = []
        self.last_id = 1

    async def get_or_create_conversation(
        self,
        session_id: str,
        agent_id: int,
        user_id=None,
        customer_profile_id=None,
        metadata=None,
    ) -> int:
        return self.last_id

    async def append_turn(self, conversation_id: int, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    async def load_history(self, conversation_id: int, limit=None):
        return list(self.turns)


class FailingConversationStore:
    async def get_or_create_conversation(self, *args, **kwargs):
        raise RuntimeError("DB unavailable")


@pytest.mark.asyncio
async def test_persistent_memory_provider_persists_turns():
    store = StubConversationStore()
    provider = PersistentMemoryProvider(store=store, max_turns=3)
    context = AgentContext(
        user=AgentUser(id=10),
        session_id="sess-1",
        metadata={"agent_id": 99},
    )

    memory = await provider.load(context)
    await memory.add_turn_async("user", "hello", {})

    assert len(store.turns) == 1
    assert store.turns[0].content == "hello"

    reloaded = await provider.load(context)
    assert any(turn.content == "hello" for turn in reloaded.history)


@pytest.mark.asyncio
async def test_persistent_memory_provider_falls_back_to_ephemeral():
    provider = PersistentMemoryProvider(
        store=FailingConversationStore(),
        max_turns=4,
        fallback=EphemeralMemoryProvider(max_turns=4),
    )
    context = AgentContext(
        user=AgentUser(id=20),
        session_id="sess-2",
        metadata={"agent_id": 101},
    )

    memory = await provider.load(context)
    assert isinstance(memory, ConversationMemory)
    assert memory.store is None
