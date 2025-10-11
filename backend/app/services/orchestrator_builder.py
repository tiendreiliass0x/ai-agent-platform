from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from functools import lru_cache
from typing import Callable, Optional

import aiohttp

from app.executor import Executor
from app.orchestrator import (
    AgentOrchestrator,
    CacheAwareContextBuilder,
    WorkingMemory,
)
from app.orchestrator.learning import LearningSystem
from app.orchestrator.memory_provider import PersistentMemoryProvider
from app.orchestrator.strategy import StrategySelector
from app.planner import Planner
from app.policy import PolicyEngine
from app.services.gemini_service import gemini_service
from app.services.tool_registry import tool_registry
from app.tooling import ToolRegistry as ToolRegistryType
from app.tooling.adapters import adapter_registry
from app.verifier import Verifier


logger = logging.getLogger("agent.orchestrator.builder")


class PermissionsRBAC:
    """Enforces permissions based on the AgentUser.permissions set."""

    def __init__(self, allow_by_default: bool = False) -> None:
        self.allow_by_default = allow_by_default

    async def has_permission(self, user, op_id: str) -> bool:
        permissions = getattr(user, "permissions", None)
        if not permissions:
            return self.allow_by_default
        if "*" in permissions:
            return True
        return op_id in permissions or f"{op_id}:execute" in permissions

    async def can_access_pii(self, user) -> bool:
        permissions = getattr(user, "permissions", None)
        if not permissions:
            return self.allow_by_default
        return any(
            perm in permissions
            for perm in {"pii:read", "pii:*", "*"}
        )


class LoggingAuditLogger:
    """Audit logger that writes gate decisions to the standard logging pipeline."""

    def __init__(self, logger_name: str = "agent.policy.audit") -> None:
        self._logger = logging.getLogger(logger_name)

    async def log_gate_decision(self, **kwargs) -> None:
        self._logger.info("policy_gate_decision", extra=kwargs)


class EnvironmentSecretStore:
    """Fetches tool credentials from environment variables."""

    def __init__(self, env=None, prefix: str = "AGENT_TOOL") -> None:
        self._env = env or os.environ
        self._prefix = prefix

    async def get_oauth_token(self, user, tool_name: str) -> str:
        return self._lookup(tool_name, "OAUTH_TOKEN")

    async def get_api_key(self, user, tool_name: str) -> str:
        return self._lookup(tool_name, "API_KEY")

    def _lookup(self, tool_name: str, suffix: str) -> str:
        key = f"{self._prefix}_{tool_name.upper().replace('.', '_')}_{suffix}"
        value = self._env.get(key)
        if value:
            return value
        raise RuntimeError(f"Missing secret: {key}")


class LoggingTelemetry:
    """Telemetry sink that records tool actions via logging."""

    def __init__(self, logger_name: str = "agent.telemetry") -> None:
        self._logger = logging.getLogger(logger_name)

    async def log_action(self, **kwargs) -> None:
        self._logger.info("tool_action", extra=kwargs)


class SimpleCircuitBreaker:
    """Basic circuit breaker with failure threshold and cooldown."""

    def __init__(self, failure_threshold: int = 5, recovery_time: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self._failure_count = 0
        self._opened_at: Optional[float] = None

    async def call(self, tool_name: str, func: Callable, *args, **kwargs):
        if self._is_open():
            raise RuntimeError(f"Circuit open for tool '{tool_name}'")

        try:
            result = await func(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        else:
            self._reset()
            return result

    def _is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if (time.monotonic() - self._opened_at) >= self.recovery_time:
            self._reset()
            return False
        return True

    def _record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._opened_at = time.monotonic()

    def _reset(self) -> None:
        self._failure_count = 0
        self._opened_at = None


class AiohttpHTTPClient:
    """HTTP client built on aiohttp with JSON response parsing."""

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def request(
        self,
        method: str,
        url: str,
        headers=None,
        json=None,
        timeout: Optional[float] = None,
    ):
        session = await self._get_session(timeout)
        start = time.perf_counter()
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
        ) as response:
            duration_ms = int((time.perf_counter() - start) * 1000)
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                payload = await response.json()
            else:
                payload = await response.text()
            return {
                "_duration_ms": duration_ms,
                "status_code": response.status,
                "data": payload,
            }

    async def _get_session(self, timeout: Optional[float]) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                client_timeout = aiohttp.ClientTimeout(total=timeout or self._timeout)
                self._session = aiohttp.ClientSession(timeout=client_timeout)
            return self._session

    async def close(self) -> None:
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None


def _default_working_memory_factory(context, task, plan):
    base = (
        context.session_id
        or context.conversation_id
        or plan.goal.replace(" ", "-")[:20]
        or uuid.uuid4().hex
    )
    slug = base or uuid.uuid4().hex
    return WorkingMemory(task_id=slug)


def create_agent_orchestrator(
    *,
    llm_service=gemini_service,
    tool_registry_instance: Optional[ToolRegistryType] = None,
    planner: Optional[Planner] = None,
    policy_engine: Optional[PolicyEngine] = None,
    executor: Optional[Executor] = None,
    strategy_selector: Optional[StrategySelector] = None,
    context_builder: Optional[CacheAwareContextBuilder] = None,
    memory_provider=None,
    learning_system=None,
    working_memory_factory=None,
    http_client=None,
    secret_store=None,
    rbac_service=None,
    audit_logger=None,
    telemetry=None,
    circuit_breaker=None,
    adapter_registry=adapter_registry,
) -> AgentOrchestrator:
    registry = tool_registry_instance or tool_registry
    planner = planner or Planner(llm_service, registry)

    rbac_service = rbac_service or PermissionsRBAC()
    audit_logger = audit_logger or LoggingAuditLogger()

    policy_engine = policy_engine or PolicyEngine(
        rbac_service=rbac_service,
        audit_logger=audit_logger,
    )

    context_builder = context_builder or CacheAwareContextBuilder(tool_registry=registry)
    strategy_selector = strategy_selector or StrategySelector()
    memory_provider = memory_provider or PersistentMemoryProvider()
    learning_system = learning_system or LearningSystem()

    http_client = http_client or AiohttpHTTPClient()
    secret_store = secret_store or EnvironmentSecretStore()
    telemetry = telemetry or LoggingTelemetry()
    circuit_breaker = circuit_breaker or SimpleCircuitBreaker()

    executor = executor or Executor(
        http_client=http_client,
        secret_store=secret_store,
        policy_engine=policy_engine,
        verifier=Verifier(),
        telemetry=telemetry,
        circuit_breaker=circuit_breaker,
        tool_registry=registry,
        adapter_registry=adapter_registry,
    )

    return AgentOrchestrator(
        planner=planner,
        llm_service=llm_service,
        executor=executor,
        tool_registry=registry,
        strategy_selector=strategy_selector,
        context_builder=context_builder,
        memory_provider=memory_provider,
        learning_system=learning_system,
        policy_engine=policy_engine,
        working_memory_factory=working_memory_factory or _default_working_memory_factory,
    )


@lru_cache(maxsize=1)
def get_agent_orchestrator() -> AgentOrchestrator:
    return create_agent_orchestrator()
