import pytest

from app.context_engine.conversation_memory import ConversationMemory
from app.orchestrator.context_builder import CacheAwareContextBuilder
from app.orchestrator.models import AgentContext, AgentTask, AgentUser
from app.tooling.models import OperationSpec, SideEffect, ToolManifest


class ToolRegistryStub:
    def __init__(self, manifests):
        self._manifests = manifests

    async def list_tools(self):
        return self._manifests


class RetrieverStub:
    async def retrieve(self, **kwargs):
        return [
            "Customer ACME has 3 open opportunities.",
            "Latest order #1234 shipped yesterday.",
        ]


def _build_manifest(name: str, description: str) -> ToolManifest:
    operations = {
        f"{name}.list": OperationSpec(
            op_id=f"{name}.list",
            method="GET",
            path="/api/list",
            side_effect=SideEffect.READ,
            description="List entities",
            args_schema={},
            returns={},
            preconditions=[],
            postconditions=[],
        ),
        f"{name}.create": OperationSpec(
            op_id=f"{name}.create",
            method="POST",
            path="/api/create",
            side_effect=SideEffect.WRITE,
            description="Create entity",
            args_schema={},
            returns={},
            preconditions=[],
            postconditions=[],
        ),
    }
    return ToolManifest(
        name=name,
        version="1.0.0",
        display_name=name.title(),
        description=description,
        auth={},
        operations=operations,
        schemas={},
        rate_limits={},
        governance={},
    )


@pytest.mark.asyncio
async def test_cache_builder_produces_stable_context():
    manifests = [
        _build_manifest("zeta.analytics", "Analytics insights"),
        _build_manifest("alpha.crm", "CRM capabilities"),
    ]
    registry = ToolRegistryStub(manifests)
    retriever = RetrieverStub()
    builder = CacheAwareContextBuilder(tool_registry=registry, knowledge_retriever=retriever)

    memory = ConversationMemory()
    memory.add_turn("user", "Need latest sales metrics.")
    memory.add_turn("assistant", "Gathering data now.")

    agent_context = AgentContext(user=AgentUser(id=1, name="Casey"))
    task = AgentTask(description="Prepare weekly revenue summary.")

    context_text = await builder.build(task, agent_context, memory)

    assert context_text.startswith(CacheAwareContextBuilder.STABLE_PREFIX)
    alpha_index = context_text.index("### alpha.crm")
    zeta_index = context_text.index("### zeta.analytics")
    assert alpha_index < zeta_index

    assert "[Context 1]" in context_text
    assert "Customer ACME has 3 open opportunities." in context_text
    assert "USER: Need latest sales metrics." in context_text
    assert context_text.strip().endswith(task.description)


@pytest.mark.asyncio
async def test_cache_builder_handles_missing_data():
    registry = ToolRegistryStub([_build_manifest("beta.support", "Support operations")])
    builder = CacheAwareContextBuilder(tool_registry=registry, knowledge_retriever=None)

    agent_context = AgentContext(user=AgentUser(id=2))
    task = AgentTask(description="Check unresolved tickets.")

    context_text = await builder.build(task, agent_context, conversation_memory=None)

    assert "## Relevant Knowledge" not in context_text
    assert "## Conversation History" not in context_text
    assert "Check unresolved tickets." in context_text
