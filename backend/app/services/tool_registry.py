from ..tooling import ToolRegistry

# Global tool registry instance used by the orchestrator and admin workflows.
tool_registry = ToolRegistry()

__all__ = ["tool_registry"]
