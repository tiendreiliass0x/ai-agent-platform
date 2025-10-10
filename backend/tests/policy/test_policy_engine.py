import pytest

from app.policy import PolicyEngine, MaskedTool
from app.policy.engine import GateResult
from app.tooling.models import OperationSpec, SideEffect, ToolManifest


class RBACStub:
    def __init__(self, permissions=None, pii=False):
        self.permissions = permissions or set()
        self.pii = pii

    async def has_permission(self, user, op_id):
        return op_id in self.permissions

    async def can_access_pii(self, user):
        return self.pii


class AuditStub:
    def __init__(self):
        self.calls = []

    async def log_gate_decision(self, **kwargs):
        self.calls.append(kwargs)


class FailureTrackerStub:
    def __init__(self, rates=None):
        self.rates = rates or {}

    async def get_failure_rate(self, op_id):
        return self.rates.get(op_id, 0.0)


@pytest.fixture
def sample_operation():
    return OperationSpec(
        op_id="crm.create_record",
        method="POST",
        path="/records",
        side_effect=SideEffect.WRITE,
        description="Create record",
        args_schema={"amount": {"type": "number"}},
        returns={},
        preconditions=[],
        postconditions=[],
    )


@pytest.fixture
def read_operation():
    return OperationSpec(
        op_id="crm.read_data",
        method="GET",
        path="/records",
        side_effect=SideEffect.READ,
        description="Read data",
        args_schema={},
        returns={},
        preconditions=[],
        postconditions=[],
    )


@pytest.fixture
def destructive_operation():
    return OperationSpec(
        op_id="crm.delete_data",
        method="DELETE",
        path="/records",
        side_effect=SideEffect.DESTRUCTIVE,
        description="Delete data",
        args_schema={},
        returns={},
        preconditions=[],
        postconditions=[],
        requires_approval=True,
    )


@pytest.mark.asyncio
async def test_gate_denies_without_permission(sample_operation):
    engine = PolicyEngine(RBACStub(), AuditStub())
    result = await engine.gate(user="alice", operation=sample_operation, args={}, tool_manifest=None)
    assert result.decision == "deny"
    assert "permissions" in result.reasoning


@pytest.mark.asyncio
async def test_gate_write_allows_with_undo(sample_operation):
    engine = PolicyEngine(RBACStub({"crm.create_record"}), AuditStub())
    result = await engine.gate(user="alice", operation=sample_operation, args={"amount": 1000}, tool_manifest=None)
    assert result.decision == "allow_with_undo"


@pytest.mark.asyncio
async def test_gate_high_risk_requires_consent(sample_operation):
    tracker = FailureTrackerStub({"crm.create_record": 1.0})
    engine = PolicyEngine(RBACStub({"crm.create_record"}), AuditStub(), failure_tracker=tracker)
    result = await engine.gate(
        user="alice",
        operation=sample_operation,
        args={"amount": 500000},
        tool_manifest=None,
    )
    assert result.decision == "require_consent"


@pytest.mark.asyncio
async def test_gate_read_with_pii_requires_permission(read_operation):
    manifest = ToolManifest(
        name="crm",
        version="1",
        display_name="CRM",
        description="CRM",
        auth={},
        operations={read_operation.op_id: read_operation},
        schemas={},
        rate_limits={},
        governance={"pii": "may_contain"},
    )

    engine = PolicyEngine(RBACStub({"crm.read_data"}, pii=False), AuditStub())
    result = await engine.gate(user="bob", operation=read_operation, args={"email": "a@b.com"}, tool_manifest=manifest)
    assert result.decision == "deny"

    engine = PolicyEngine(RBACStub({"crm.read_data"}, pii=True), AuditStub())
    result = await engine.gate(user="bob", operation=read_operation, args={"email": "a@b.com"}, tool_manifest=manifest)
    assert result.decision == "allow"


@pytest.mark.asyncio
async def test_gate_destructive_requires_consent(destructive_operation):
    engine = PolicyEngine(RBACStub({"crm.delete_data"}), AuditStub())
    result = await engine.gate(user="alice", operation=destructive_operation, args={}, tool_manifest=None)
    assert result.decision == "require_consent"


def test_mask_tools_and_logit_bias(sample_operation):
    manifest = ToolManifest(
        name="crm",
        version="v1",
        display_name="CRM",
        description="",
        auth={},
        operations={
            sample_operation.op_id: sample_operation,
            "crm.other": sample_operation,
        },
        schemas={},
        rate_limits={},
        governance={},
    )
    engine = PolicyEngine(RBACStub({"crm.create_record"}), AuditStub())
    masked = list(engine.mask_tools([manifest], {"crm.create_record"}))
    assert masked[0].visible is True
    assert masked[0].visible_operations == {"crm.create_record"}

    bias = engine.get_logit_bias(masked)
    assert bias.get("crm.other") == -100.0
