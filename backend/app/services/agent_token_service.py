"""Utility helpers for issuing and validating short-lived agent session tokens."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt

from app.core.config import settings


class AgentTokenError(Exception):
    """Raised when agent session token validation fails."""


def create_agent_session_token(agent_public_id: str, ttl_seconds: int = 300) -> str:
    """Create a signed, short-lived token for an agent session.

    Args:
        agent_public_id: Public UUID identifying the agent.
        ttl_seconds: Token lifetime (default 5 minutes).

    Returns:
        Signed JWT string that encodes agent identity and expiry.
    """

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    payload = {
        "sub": "agent_session",
        "agent_public_id": agent_public_id,
        "exp": expires_at,
        "iat": datetime.now(timezone.utc),
    }

    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def verify_agent_session_token(token: str) -> str:
    """Validate an agent session token and return the agent public id.

    Raises AgentTokenError if invalid or expired.
    """

    try:
        decoded = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AgentTokenError("Session token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AgentTokenError("Invalid session token") from exc

    agent_public_id = decoded.get("agent_public_id")
    if not agent_public_id:
        raise AgentTokenError("Session token missing agent identifier")

    return agent_public_id
