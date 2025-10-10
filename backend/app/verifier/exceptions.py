class VerifierError(Exception):
    """Base class for verifier errors."""


class ConditionEvaluationError(VerifierError):
    """Raised when a condition expression cannot be evaluated safely."""
