import jwt
import pytest

from app.services.agent_token_service import (
    create_agent_session_token,
    verify_agent_session_token,
    AgentTokenError,
)
from app.core.config import settings


def test_agent_session_token_roundtrip():
    public_id = "123e4567-e89b-12d3-a456-426614174000"
    token = create_agent_session_token(public_id, ttl_seconds=30)

    decoded_public_id = verify_agent_session_token(token)

    assert decoded_public_id == public_id


def test_agent_session_token_expired():
    token = create_agent_session_token("123e4567-e89b-12d3-a456-426614174000", ttl_seconds=-1)

    with pytest.raises(AgentTokenError):
        verify_agent_session_token(token)


def test_agent_session_token_missing_agent_id():
    token = jwt.encode({"sub": "agent_session"}, settings.SECRET_KEY, algorithm="HS256")

    with pytest.raises(AgentTokenError):
        verify_agent_session_token(token)
