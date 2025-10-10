class ToolExecutionError(Exception):
    """Raised when tool execution fails after retries."""


class PreconditionError(ToolExecutionError):
    """Raised when preconditions fail."""


class PostconditionError(ToolExecutionError):
    """Raised when postconditions fail."""
