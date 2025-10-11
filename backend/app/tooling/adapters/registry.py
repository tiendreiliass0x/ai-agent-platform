from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Optional

from .base import ToolAdapter, ToolExecutionContext


AdapterFactory = Callable[[Dict[str, Any]], ToolAdapter]


class AdapterRegistry:
    """Registry responsible for returning adapters for a given tool."""

    def __init__(self) -> None:
        self._adapters: Dict[str, list[AdapterFactory]] = defaultdict(list)

    def register(self, tool_name: str, factory: AdapterFactory) -> None:
        self._adapters[tool_name].append(factory)

    def get_adapter(self, tool_name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[ToolAdapter]:
        candidates = [tool_name]
        if "." in tool_name:
            candidates.append(tool_name.split(".", 1)[0])

        factories = None
        for name in candidates:
            factories = self._adapters.get(name)
            if factories:
                break
        if not factories:
            return None
        metadata = metadata or {}
        for factory in factories:
            adapter = factory(metadata)
            if adapter is not None:
                return adapter
        return None


adapter_registry = AdapterRegistry()
