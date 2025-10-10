import pytest

from app.executor import Executor, ToolExecutionError, PreconditionError, PostconditionError
from app.tooling.models import OperationSpec, SideEffect, ToolManifest
from app.policy.engine import GateResult
from app.planner.models import Step
from app.verifier import Verifier


class PolicyStub:
    def __init__(self, decision="allow"):
        self.decision = decision

    async def gate(self, user, operation, args, manifest):
        return GateResult(decision=self.decision, reasoning="stub", risk_score=0.0)


class PolicyDenyStub(PolicyStub):
    def __init__(self):
        super().__init__(decision="deny")


class TelemetryStub:
    def __init__(self):
        self.calls = []

    async def log_action(self, **kwargs):
        self.calls.append(kwargs)


class CircuitBreakerStub:
    async def call(self, tool_name, func, *args, **kwargs):
        return await func(*args, **kwargs)


class SecretStoreStub:
    async def get_oauth_token(self, user, tool_name):
        return "token"

    async def get_api_key(self, user, tool_name):
        return "api-key"


class HTTPStub:
    def __init__(self, response=None, error=None):
        self.response = response or {"_duration_ms": 10, "status": "ok"}
        self.error = error
        self.calls = []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return dict(self.response)


class HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class RegistryStub:
    def __init__(self, manifest):
        self.manifest = manifest

    async def get_tool(self, name, version=None):
        return self.manifest


def make_manifest(pre=None, post=None, compensation=None):
    operation = OperationSpec(
        op_id="crm.create",
        method="POST",
        path="/create",
        side_effect=SideEffect.WRITE,
        description="",
        args_schema={"value": {"type": "number"}},
        returns={},
        preconditions=pre or [],
        postconditions=post or [],
        idempotency_header="Idempotency-Key",
        requires_approval=False,
        compensation=compensation,
        errors=[{"code": 429, "retry": "exponential_backoff"}],
    )
    manifest = ToolManifest(
        name="crm",
        version="1.0",
        display_name="CRM",
        description="",
        auth={"kind": "oauth2"},
        operations={operation.op_id: operation},
        schemas={},
        rate_limits={},
        governance={},
        raw={"base_url": "https://example.com"},
    )
    return manifest


@pytest.mark.asyncio
async def test_executor_success():
    manifest = make_manifest(pre=["args['value'] > 0"], post=["result['status'] == 'ok'"])
    registry = RegistryStub(manifest)
    telemetry = TelemetryStub()
    http_stub = HTTPStub()
    executor = Executor(
        http_client=http_stub,
        secret_store=SecretStoreStub(),
        policy_engine=PolicyStub(),
        verifier=Verifier(),
        telemetry=telemetry,
        circuit_breaker=CircuitBreakerStub(),
        tool_registry=registry,
    )

    user = type("User", (), {"id": 1})()
    step = Step(id="step1", kind="tool", action="crm.create", args={"value": 10})
    result = await executor.execute(step, user=user, context={})

    assert result["status"] == "ok"
    assert telemetry.calls
    headers = http_stub.calls[0]["headers"]
    assert headers["Authorization"] == "Bearer token"
    assert "Idempotency-Key" in headers


@pytest.mark.asyncio
async def test_executor_policy_denied():
    executor = Executor(
        http_client=HTTPStub(),
        secret_store=SecretStoreStub(),
        policy_engine=PolicyDenyStub(),
        verifier=Verifier(),
        telemetry=TelemetryStub(),
        circuit_breaker=CircuitBreakerStub(),
        tool_registry=RegistryStub(make_manifest()),
    )

    step = Step(id="step1", kind="tool", action="crm.create", args={"value": 1})
    with pytest.raises(ToolExecutionError):
        await executor.execute(step, user=object(), context={})


@pytest.mark.asyncio
async def test_executor_precondition_failure():
    executor = Executor(
        http_client=HTTPStub(),
        secret_store=SecretStoreStub(),
        policy_engine=PolicyStub(),
        verifier=Verifier(),
        telemetry=TelemetryStub(),
        circuit_breaker=CircuitBreakerStub(),
        tool_registry=RegistryStub(make_manifest(pre=["args['value'] > 0"])),
    )

    step = Step(id="step1", kind="tool", action="crm.create", args={"value": -1})
    with pytest.raises(PreconditionError):
        await executor.execute(step, user=object(), context={})


@pytest.mark.asyncio
async def test_executor_postcondition_failure():
    executor = Executor(
        http_client=HTTPStub(response={"status": "error"}),
        secret_store=SecretStoreStub(),
        policy_engine=PolicyStub(),
        verifier=Verifier(),
        telemetry=TelemetryStub(),
        circuit_breaker=CircuitBreakerStub(),
        tool_registry=RegistryStub(make_manifest(post=["result['status'] == 'ok'"])),
    )

    step = Step(id="step1", kind="tool", action="crm.create", args={"value": 5})
    with pytest.raises(PostconditionError):
        await executor.execute(step, user=object(), context={})


@pytest.mark.asyncio
async def test_executor_retry_exhausted():
    executor = Executor(
        http_client=HTTPStub(error=HTTPError(500)),
        secret_store=SecretStoreStub(),
        policy_engine=PolicyStub(),
        verifier=Verifier(),
        telemetry=TelemetryStub(),
        circuit_breaker=CircuitBreakerStub(),
        tool_registry=RegistryStub(make_manifest()),
    )

    step = Step(id="step1", kind="tool", action="crm.create", args={"value": 5})
    with pytest.raises(ToolExecutionError):
        await executor.execute(step, user=object(), context={})
