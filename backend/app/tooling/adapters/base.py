from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from app.tooling.models import OperationSpec, ToolManifest


@dataclass
class ToolExecutionContext:
    """Execution context provided to adapters."""

    user: Any
    shared_state: Dict[str, Any]
    agent_metadata: Dict[str, Any]
    secret_store: Any
    http_client: Any


class ToolAdapter(Protocol):
    """Adapter interface for tool executions."""

    async def execute(
        self,
        manifest: ToolManifest,
        operation: OperationSpec,
        args: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> Dict[str, Any]:
        ...
