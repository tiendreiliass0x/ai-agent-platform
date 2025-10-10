class PolicyError(Exception):
    """Base exception for policy engine errors."""


class RBACPermissionError(PolicyError):
    """Raised when RBAC denies an operation."""
