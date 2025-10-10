import pytest

from app.orchestrator import AgentOrchestrator, AgentTask, AgentContext, AgentUser
from app.planner.models import Step, Plan
from app.policy.engine import GateResult
from app.executor import ToolExecutionError
from app.tooling.models import ToolManifest, OperationSpec, SideEffect
from app.context_engine.conversation_memory import ConversationMemory
from app.reasoning import ReasoningEngine
from app.reasoning.engine import BaseReasoner


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
    async def select(self, task, context):
        return type("Decision", (), {"name": "plan_execute", "reasoning": "stub"})()


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
    assert "policy denied" in result.error
    assert learner.records[-1] is False
