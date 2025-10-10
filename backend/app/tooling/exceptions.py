class ToolRegistryError(Exception):
    """Base exception for tool registry issues."""


class ManifestValidationError(ToolRegistryError):
    """Raised when a tool manifest fails validation."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not registered."""


class OperationNotFoundError(ToolRegistryError):
    """Raised when a requested operation is absent."""
