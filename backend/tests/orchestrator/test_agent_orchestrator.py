import pytest

from app.orchestrator import AgentOrchestrator, AgentTask, AgentContext, AgentUser
from app.planner.models import Step, Plan
from app.policy.engine import GateResult
from app.executor import ToolExecutionError
from app.tooling.models import ToolManifest, OperationSpec, SideEffect
from app.context_engine.conversation_memory import ConversationMemory
from app.reasoning import ReasoningEngine
from app.reasoning.engine import BaseReasoner
from app.tooling.adapters.registry import AdapterRegistry
from app.executor import Executor
from app.verifier import Verifier


class ReasonerStub(BaseReasoner):
    def __init__(self, plan: Plan):
        self.plan = plan

    async def generate_plan(self, **kwargs):
        return self.plan


class ExecutorStub:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def execute(self, step, user, context):
        self.calls.append((step, context))
        if isinstance(self.responses, Exception):
            raise self.responses
        return self.responses[step.id]


class MemoryProviderStub:
    def __init__(self):
        self.memory = ConversationMemory()
        self.saved = False

    async def load(self, context):
        return self.memory

    async def save(self, context, memory):
        self.saved = True


class StrategySelectorStub:
    def __init__(self, name: str = "plan_execute"):
        self.name = name

    async def select(self, task, context):
        return type("Decision", (), {"name": self.name, "reasoning": "stub"})()


class ContextBuilderStub:
    async def build(self, task, context, memory):
        return "context"


class LearningSystemStub:
    def __init__(self):
        self.records = []

    async def record_execution(self, plan, success: bool):
        self.records.append(success)


class PolicyEngineStub:
    def mask_tools(self, tools, permissions):
        return []

    def get_logit_bias(self, masked):
        return {}


class ToolRegistryStub:
    async def list_tools(self):
        return []


class ReflexionReasonerStub(BaseReasoner):
    def __init__(self):
        self.calls = []

    async def generate_plan(self, **kwargs):
        self.calls.append(kwargs)
        step_id = "step_retry"
        plan = Plan(
            goal="Recover",
            strategy="reflexion",
            steps=[Step(id=step_id, kind="tool", action="crm.fix", args={})],
        )
        return plan


class FlakyExecutorStub:
    def __init__(self):
        self.calls = 0

    async def execute(self, step, user, context):
        self.calls += 1
        if self.calls == 1:
            raise ToolExecutionError("transient failure")
        return {"status": "ok"}


class AdapterStub:
    def __init__(self):
        self.calls = 0

    async def execute(self, manifest, operation, args, context):
        self.calls += 1
        return {"status": "adapter"}


class HTTPClientStub:
    def __init__(self, response=None):
        self.response = response or {"_duration_ms": 5, "status": "ok"}
        self.calls = []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.response)


class SecretStoreStub:
    async def get_oauth_token(self, user, tool_name):
        return "token"

    async def get_api_key(self, user, tool_name):
        return "key"


class PolicyStub:
    async def gate(self, user, operation, args, manifest):
        return GateResult(decision="allow", reasoning="stub", risk_score=0.0)


class TelemetryStub:
    def __init__(self):
        self.calls = []

    async def log_action(self, **kwargs):
        self.calls.append(kwargs)


class CircuitBreakerStub:
    async def call(self, tool_name, func, *args, **kwargs):
        return await func(*args, **kwargs)


@pytest.mark.asyncio
async def test_orchestrator_executes_plan_successfully():
    plan = Plan(
        goal="Demo",
        strategy="plan_execute",
        steps=[
            Step(id="step_1", kind="tool", action="crm.search", args={"q": "acme"}),
            Step(
                id="step_2",
                kind="tool",
                action="crm.create",
                args={"account_id": "$step_1.result['id']"},
                depends_on=["step_1"],
            ),
        ],
    )

    reasoners = {"plan_execute": ReasonerStub(plan)}
    reasoning_engine = ReasoningEngine(reasoners)
    executor = ExecutorStub({
        "step_1": {"id": "ACC-1"},
        "step_2": {"status": "created"},
    })
    memory_provider = MemoryProviderStub()

    learner = LearningSystemStub()
    orchestrator = AgentOrchestrator(
        reasoning_engine=reasoning_engine,
        executor=executor,
        tool_registry=ToolRegistryStub(),
        strategy_selector=StrategySelectorStub(),
        context_builder=ContextBuilderStub(),
        memory_provider=memory_provider,
        learning_system=learner,
        policy_engine=PolicyEngineStub(),
    )

    context = AgentContext(user=AgentUser(id=1))
    task = AgentTask(description="Create opportunity")
    result = await orchestrator.run_task(context, task)

    assert result.status == "completed"
    assert result.step_results["step_2"]["status"] == "completed"
    assert result.step_results["step_2"]["result"]["status"] == "created"
    assert memory_provider.saved is True
    assert learner.records[-1] is True


@pytest.mark.asyncio
async def test_orchestrator_handles_execution_failure():
    plan = Plan(
        goal="Demo",
        strategy="plan_execute",
        steps=[Step(id="step_1", kind="tool", action="crm.op", args={})],
    )

    reasoning_engine = ReasoningEngine({"plan_execute": ReasonerStub(plan)})
    executor = ExecutorStub(ToolExecutionError("policy denied"))
    learner = LearningSystemStub()
    orchestrator = AgentOrchestrator(
        reasoning_engine=reasoning_engine,
        executor=executor,
        tool_registry=ToolRegistryStub(),
        strategy_selector=StrategySelectorStub(),
        context_builder=ContextBuilderStub(),
        memory_provider=MemoryProviderStub(),
        learning_system=learner,
        policy_engine=PolicyEngineStub(),
    )

    context = AgentContext(user=AgentUser(id=1))
    task = AgentTask(description="Denied task")
    result = await orchestrator.run_task(context, task)

    assert result.status == "failed"
    assert result.step_results["step_1"]["status"] == "failed"
    assert "policy denied" in result.error
    assert learner.records[-1] is False


@pytest.mark.asyncio
async def test_orchestrator_reflexion_retries_on_failure():
    reasoner = ReflexionReasonerStub()
    reasoning_engine = ReasoningEngine({"reflexion": reasoner})
    executor = FlakyExecutorStub()
    learner = LearningSystemStub()
    memory_provider = MemoryProviderStub()

    orchestrator = AgentOrchestrator(
        reasoning_engine=reasoning_engine,
        executor=executor,
        tool_registry=ToolRegistryStub(),
        strategy_selector=StrategySelectorStub(name="reflexion"),
        context_builder=ContextBuilderStub(),
        memory_provider=memory_provider,
        learning_system=learner,
        policy_engine=PolicyEngineStub(),
    )

    context = AgentContext(user=AgentUser(id=99))
    task = AgentTask(description="Recover from failure")
    result = await orchestrator.run_task(context, task)

    assert result.status == "completed"
    assert learner.records == [False, True]
    assert len(reasoner.calls) >= 2
    assert reasoner.calls[1]["feedback"]
    assert "step_retry" in reasoner.calls[1]["feedback"]


@pytest.mark.asyncio
async def test_executor_uses_tool_adapter():
    manifest = ToolManifest(
        name="crm",
        version="1.0",
        display_name="CRM",
        description="",
        auth={},
        operations={
            "crm.create": OperationSpec(
                op_id="crm.create",
                method="POST",
                path="/create",
                side_effect=SideEffect.WRITE,
                description="",
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

    class ManifestRegistry:
        async def get_tool(self, name, version=None):
            return manifest

        async def list_tools(self):
            return [manifest]

    plan = Plan(
        goal="Adapter demo",
        strategy="plan_execute",
        steps=[Step(id="step_1", kind="tool", action="crm.create", args={})],
    )

    reasoning_engine = ReasoningEngine({"plan_execute": ReasonerStub(plan)})
    adapter_stub = AdapterStub()
    registry = AdapterRegistry()

    def factory(metadata):
        if metadata.get("integrations", {}).get("crm", {}).get("adapter") == "salesforce":
            return adapter_stub
        return None

    registry.register("crm", factory)

    executor = Executor(
        http_client=HTTPClientStub(),
        secret_store=SecretStoreStub(),
        policy_engine=PolicyStub(),
        verifier=Verifier(),
        telemetry=TelemetryStub(),
        circuit_breaker=CircuitBreakerStub(),
        tool_registry=ManifestRegistry(),
        adapter_registry=registry,
    )

    orchestrator = AgentOrchestrator(
        reasoning_engine=reasoning_engine,
        executor=executor,
        tool_registry=ManifestRegistry(),
        strategy_selector=StrategySelectorStub(),
        context_builder=ContextBuilderStub(),
        memory_provider=MemoryProviderStub(),
        learning_system=LearningSystemStub(),
        policy_engine=PolicyEngineStub(),
    )

    context = AgentContext(
        user=AgentUser(id=1),
        metadata={"integrations": {"crm": {"adapter": "salesforce"}}},
    )
    task = AgentTask(description="Use adapter")
    result = await orchestrator.run_task(context, task)

    assert result.status == "completed"
    assert adapter_stub.calls == 1
