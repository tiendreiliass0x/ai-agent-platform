from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Set

from app.tooling.models import OperationSpec, ToolManifest, SideEffect
from .models import MaskedTool
from .exceptions import PolicyError

Decision = Literal["allow", "allow_with_undo", "require_consent", "deny"]


@dataclass
class GateResult:
    decision: Decision
    reasoning: str
    risk_score: float


class PolicyEngine:
    """Risk-aware gating for tool executions."""

    def __init__(
        self,
        rbac_service,
        audit_logger,
        pii_detector=None,
        failure_tracker=None,
    ) -> None:
        self._rbac = rbac_service
        self._audit = audit_logger
        self._pii_detector = pii_detector
        self._failure_tracker = failure_tracker

    async def gate(
        self,
        user: Any,
        operation: OperationSpec,
        args: Dict[str, Any],
        tool_manifest: Optional[ToolManifest] = None,
    ) -> GateResult:
        if not await self._rbac.has_permission(user, operation.op_id):
            result = GateResult(
                decision="deny",
                reasoning="User lacks required permissions",
                risk_score=0.0,
            )
            await self._log_decision(user, operation, args, result)
            return result

        risk_score = await self._calculate_risk(operation, args)

        if getattr(operation, "requires_approval", False):
            result = GateResult(
                decision="require_consent",
                reasoning="Operation requires explicit approval",
                risk_score=risk_score,
            )
            await self._log_decision(user, operation, args, result)
            return result

        side_effect = operation.side_effect

        if side_effect == SideEffect.DESTRUCTIVE:
            decision: Decision = "require_consent"
            reasoning = "Destructive operations require user confirmation"
        elif side_effect == SideEffect.WRITE:
            if risk_score > 0.7:
                decision = "require_consent"
                reasoning = f"High risk score ({risk_score:.2f}) requires approval"
            else:
                decision = "allow_with_undo"
                reasoning = "Write operation permitted with undo capability"
        elif side_effect == SideEffect.READ:
            if self._contains_pii(operation, args, tool_manifest):
                if not await self._rbac.can_access_pii(user):
                    decision = "deny"
                    reasoning = "User cannot access PII"
                else:
                    decision = "allow"
                    reasoning = "Read operation with PII access granted"
            else:
                decision = "allow"
                reasoning = "Read operation permitted"
        else:
            decision = "deny"
            reasoning = "Unknown operation type"

        result = GateResult(decision=decision, reasoning=reasoning, risk_score=risk_score)
        await self._log_decision(user, operation, args, result)
        return result

    def mask_tools(
        self,
        all_tools: Iterable[ToolManifest],
        user_permissions: Set[str],
    ) -> List[MaskedTool]:
        masked: list[MaskedTool] = []
        permission_set = set(user_permissions)

        for manifest in all_tools:
            operation_ids = set(manifest.operations.keys())
            visible_ops = operation_ids & permission_set
            masked.append(
                MaskedTool(
                    manifest=manifest,
                    visible=bool(visible_ops),
                    visible_operations=visible_ops,
                )
            )
        return masked

    def get_logit_bias(self, masked_tools: Iterable[MaskedTool]) -> Dict[str, float]:
        bias: Dict[str, float] = {}
        for entry in masked_tools:
            if not entry.visible:
                bias[entry.manifest.name] = -100.0
                for op_id in entry.manifest.operations.keys():
                    bias[op_id] = -100.0
                continue
            for op_id in entry.manifest.operations.keys():
                if op_id not in entry.visible_operations:
                    bias[op_id] = -100.0
        return bias

    async def _calculate_risk(self, operation: OperationSpec, args: Dict[str, Any]) -> float:
        risk = 0.0

        if operation.side_effect == SideEffect.READ:
            risk += 0.1
        elif operation.side_effect == SideEffect.WRITE:
            risk += 0.4
        elif operation.side_effect == SideEffect.DESTRUCTIVE:
            risk += 0.8

        amount = args.get("amount")
        if isinstance(amount, (int, float)):
            if amount > 100_000:
                risk += 0.3
            elif amount > 10_000:
                risk += 0.2

        items = args.get("items")
        if isinstance(items, list) and len(items) > 100:
            risk += 0.2

        failure_rate = await self._get_failure_rate(operation.op_id)
        risk += failure_rate * 0.2

        return min(risk, 1.0)

    async def _get_failure_rate(self, op_id: str) -> float:
        if self._failure_tracker is None:
            return 0.0
        value = await self._failure_tracker.get_failure_rate(op_id)
        return max(0.0, min(float(value), 1.0))

    def _contains_pii(
        self,
        operation: OperationSpec,
        args: Dict[str, Any],
        tool_manifest: Optional[ToolManifest],
    ) -> bool:
        if self._pii_detector is not None:
            return self._pii_detector.contains_pii(args)
        if not tool_manifest:
            return False
        governance = tool_manifest.governance or {}
        pii_flag = governance.get("pii")
        return pii_flag in {"may_contain", "contains"} and bool(args)

    async def _log_decision(
        self,
        user: Any,
        operation: OperationSpec,
        args: Dict[str, Any],
        result: GateResult,
    ) -> None:
        if hasattr(self._audit, "log_gate_decision"):
            await self._audit.log_gate_decision(
                user=user,
                operation=operation,
                args=args,
                decision=result.decision,
                risk_score=result.risk_score,
                reasoning=result.reasoning,
            )
