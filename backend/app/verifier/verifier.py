from __future__ import annotations

from typing import Any, Dict, Iterable, List

from app.tooling.models import OperationSpec
from .exceptions import VerifierError
from .safe_eval import safe_eval


class Verifier:
    """Evaluates pre and postconditions defined in tool manifests."""

    def verify_preconditions(self, operation: OperationSpec, args: Dict[str, Any]) -> None:
        self._evaluate_conditions(operation.preconditions, {"args": args})

    def verify_postconditions(
        self,
        operation: OperationSpec,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        self._evaluate_conditions(operation.postconditions, {"args": args, "result": result})

    def _evaluate_conditions(self, conditions: Iterable[str], context: Dict[str, Any]) -> None:
        for condition in conditions or []:
            if not condition:
                continue
            value = safe_eval(condition, context)
            if not value:
                raise VerifierError(f"Condition failed: {condition}")
