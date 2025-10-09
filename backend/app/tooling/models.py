from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SideEffect(str, Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass
class OperationSpec:
    op_id: str
    method: str
    path: str
    side_effect: SideEffect
    description: str = ""
    args_schema: Dict[str, Any] = field(default_factory=dict)
    returns: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    idempotency_header: Optional[str] = None
    requires_approval: bool = False
    compensation: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolManifest:
    name: str
    version: str
    display_name: str
    description: str
    auth: Dict[str, Any]
    operations: Dict[str, OperationSpec]
    schemas: Dict[str, Any] = field(default_factory=dict)
    rate_limits: Dict[str, Any] = field(default_factory=dict)
    governance: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def get_operation(self, op_id: str) -> OperationSpec:
        try:
            return self.operations[op_id]
        except KeyError as exc:
            raise KeyError(f"Operation '{op_id}' not found for tool '{self.name}'.") from exc
