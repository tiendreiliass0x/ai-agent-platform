from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from app.planner.models import Plan


@dataclass
class AgentUser:
    id: Any
    name: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)


@dataclass
class AgentContext:
    user: AgentUser
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    status: str
    plan: Plan
    step_results: Dict[str, Dict[str, Any]]
    error: Optional[str] = None
