import json
import pytest

from app.planner import Planner, PlanValidationError
from app.tooling.models import OperationSpec, SideEffect, ToolManifest


class LLMStub:
    def __init__(self, response: str):
        self.response = response
        self.prompt = None

    async def generate_response(self, **kwargs):
        self.prompt = kwargs.get("prompt")
        return self.response


class RegistryStub:
    def __init__(self, manifests):
        self.manifests = manifests

    async def search_tools(self, description, top_k=10):
        return self.manifests


@pytest.fixture
def tool_manifest():
    operations = {
        "crm.search_contacts": OperationSpec(
            op_id="crm.search_contacts",
            method="GET",
            path="/contacts/search",
            side_effect=SideEffect.READ,
            description="Search contacts",
            args_schema={},
            returns={},
            preconditions=[],
            postconditions=[],
        ),
        "crm.create_opportunity": OperationSpec(
            op_id="crm.create_opportunity",
            method="POST",
            path="/opportunities",
            side_effect=SideEffect.WRITE,
            description="Create opportunity",
            args_schema={},
            returns={},
            preconditions=[],
            postconditions=[],
        ),
    }
    return ToolManifest(
        name="salesforce.crm",
        version="1.0.0",
        display_name="Salesforce",
        description="CRM",
        auth={},
        operations=operations,
        schemas={},
        rate_limits={},
        governance={},
    )


@pytest.mark.asyncio
async def test_planner_generates_plan(tool_manifest):
    plan_json = json.dumps(
        {
            "goal": "Create opportunity",
            "steps": [
                {
                    "id": "step_1",
                    "kind": "tool",
                    "action": "crm.search_contacts",
                    "args": {"q": "ACME"},
                    "depends_on": [],
                    "reasoning": "find contact",
                },
                {
                    "id": "step_2",
                    "kind": "tool",
                    "action": "crm.create_opportunity",
                    "args": {"name": "ACME"},
                    "depends_on": ["step_1"],
                    "reasoning": "create opportunity",
                },
            ],
            "parallel_groups": [],
        }
    )
    planner = Planner(LLMStub(plan_json), RegistryStub([tool_manifest]))

    class Task:
        description = "Create opportunity for ACME"

    plan = await planner.plan(Task(), strategy="plan_execute", context="")
    assert plan.goal == "Create opportunity"
    assert len(plan.steps) == 2
    assert plan.steps[1].depends_on == ["step_1"]


@pytest.mark.asyncio
async def test_planner_invalid_tool_raises(tool_manifest):
    bad_json = json.dumps(
        {
            "goal": "Bad",
            "steps": [
                {
                    "id": "step_1",
                    "kind": "tool",
                    "action": "unknown.tool",
                    "args": {},
                    "depends_on": [],
                }
            ],
        }
    )
    planner = Planner(LLMStub(bad_json), RegistryStub([tool_manifest]))

    class Task:
        description = "Test"

    with pytest.raises(PlanValidationError):
        await planner.plan(Task(), strategy="plan", context="")


@pytest.mark.asyncio
async def test_planner_detects_cycles(tool_manifest):
    cyclic_plan = json.dumps(
        {
            "goal": "Cycle",
            "steps": [
                {
                    "id": "a",
                    "kind": "tool",
                    "action": "crm.search_contacts",
                    "args": {},
                    "depends_on": ["b"],
                },
                {
                    "id": "b",
                    "kind": "tool",
                    "action": "crm.create_opportunity",
                    "args": {},
                    "depends_on": ["a"],
                },
            ],
        }
    )
    planner = Planner(LLMStub(cyclic_plan), RegistryStub([tool_manifest]))

    class Task:
        description = "Cycle"

    with pytest.raises(PlanValidationError):
        await planner.plan(Task(), strategy="plan", context="")
