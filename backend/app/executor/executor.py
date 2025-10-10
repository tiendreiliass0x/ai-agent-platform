from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from typing import Any, Dict, Optional

from jsonschema import validate as json_validate

from app.tooling.models import ToolManifest, OperationSpec
from app.policy import PolicyEngine
from app.verifier import Verifier, VerifierError
from .exceptions import PreconditionError, PostconditionError, ToolExecutionError
from .sagas import SagaManager, SagaStep


class Executor:
    def __init__(
        self,
        http_client,
        secret_store,
        policy_engine: PolicyEngine,
        verifier: Verifier,
        telemetry,
        circuit_breaker,
        tool_registry,
        saga_manager: SagaManager | None = None,
    ) -> None:
        self.http = http_client
        self.secrets = secret_store
        self.policy = policy_engine
        self.telemetry = telemetry
        self.circuit_breaker = circuit_breaker
        self.registry = tool_registry
        self.saga_manager = saga_manager or SagaManager()
        self.verifier = verifier

    async def execute(self, step: Any, user: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if step.kind != "tool":
            raise ValueError("Executor can only handle tool steps")

        tool_name, op_id = step.action.split(".", 1)
        manifest: ToolManifest = await self.registry.get_tool(tool_name)
        operation: OperationSpec = manifest.operations[step.action]

        gate = await self.policy.gate(user, operation, step.args, manifest)
        if gate.decision == "deny":
            raise ToolExecutionError(gate.reasoning)
        if gate.decision == "require_consent":
            raise ToolExecutionError("Consent required before executing operation")

        self._validate_args(step.args, operation)
        try:
            self.verifier.verify_preconditions(operation, step.args)
        except VerifierError as exc:
            raise PreconditionError(str(exc)) from exc

        headers = await self._build_headers(user, manifest, operation, step.args)

        async def invoke():
            return await self._execute_with_retries(manifest, operation, step.args, headers)

        response = await self.circuit_breaker.call(manifest.name, invoke)
        try:
            self.verifier.verify_postconditions(operation, step.args, response)
        except VerifierError as exc:
            raise PostconditionError(str(exc)) from exc

        await self.telemetry.log_action(
            user=user,
            tool=manifest.name,
            operation=step.action,
            args=step.args,
            result=response,
            duration_ms=response.get("_duration_ms", 0),
        )

        if operation.compensation:
            self.saga_manager.add_step(
                SagaStep(
                    rollback_operation=operation.compensation.get("operation"),
                    args=operation.compensation.get("args_mapping", {}),
                )
            )

        return response

    def _validate_args(self, args: Dict[str, Any], operation: OperationSpec) -> None:
        schema = operation.args_schema or {}
        if schema:
            json_validate(instance=args, schema={"type": "object", "properties": schema})

    async def _build_headers(
        self,
        user: Any,
        tool: ToolManifest,
        operation: OperationSpec,
        args: Dict[str, Any],
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}

        auth = tool.auth or {}
        kind = auth.get("kind")
        if kind == "oauth2":
            token = await self.secrets.get_oauth_token(user, tool.name)
            headers["Authorization"] = f"Bearer {token}"
        elif kind == "api_key":
            api_key = await self.secrets.get_api_key(user, tool.name)
            headers["X-API-Key"] = api_key

        if operation.idempotency_header:
            headers[operation.idempotency_header] = self._generate_idempotency_key(
                getattr(user, "id", "user"),
                operation.op_id,
                args,
            )

        return headers

    def _generate_idempotency_key(self, user_id: Any, op_id: str, args: Dict[str, Any]) -> str:
        args_json = json.dumps(args, sort_keys=True)
        args_hash = hashlib.sha256(args_json.encode()).hexdigest()[:12]
        bucket = int(time.time() // 600)
        return f"{user_id}:{op_id}:{args_hash}:{bucket}"

    async def _execute_with_retries(
        self,
        tool: ToolManifest,
        operation: OperationSpec,
        args: Dict[str, Any],
        headers: Dict[str, str],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        attempt = 0
        last_error: Exception | None = None

        while attempt < max_retries:
            try:
                start = time.time()
                response = await self.http.request(
                    method=operation.method,
                    url=self._build_url(tool, operation, args),
                    headers=headers,
                    json=args,
                    timeout=20.0,
                )
                response["_duration_ms"] = int((time.time() - start) * 1000)
                return response
            except Exception as exc:  # replace with HTTPError in integration
                last_error = exc
                if not self._should_retry(exc, operation):
                    raise
                backoff = (2 ** attempt) + random.random() * 0.1
                await asyncio.sleep(backoff)
                attempt += 1

        raise ToolExecutionError(f"Failed after {max_retries} attempts: {last_error}")

    def _build_url(self, tool: ToolManifest, operation: OperationSpec, args: Dict[str, Any]) -> str:
        base_url = tool.raw.get("base_url") if tool.raw else None
        if base_url:
            return f"{base_url}{operation.path}"
        return operation.path

    def _should_retry(self, error: Exception, operation: OperationSpec) -> bool:
        for spec in operation.errors or []:
            if getattr(error, "status_code", None) == spec.get("code"):
                return spec.get("retry") == "exponential_backoff"
        status = getattr(error, "status_code", None)
        return status in {429} or (status is not None and status >= 500)

    async def rollback(self, executor_callable) -> None:
        await self.saga_manager.rollback(executor_callable)
