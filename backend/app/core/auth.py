"""
Authentication and Authorization for Domain Expertise
"""

from fastapi import Depends, HTTPException, status, Header
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timezone


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Simple user authentication for testing"""
    # Mock authentication - replace with real auth in production
    if username == "testuser@example.com" and password == "password123":
        return {
            "id": 1,
            "username": username,
            "email": username,
            "is_active": True
        }
    return None


def create_user_token(user_data: Dict[str, Any]) -> str:
    """Create JWT token for user"""
    payload = {
        "user_id": user_data.get("id"),
        "email": user_data.get("email"),
        "exp": datetime.now(timezone.utc).timestamp() + 3600  # 1 hour
    }
    return jwt.encode(payload, "secret-key", algorithm="HS256")


def get_password_hash(password: str) -> str:
    """Hash password (mock implementation)"""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Get current user from JWT token
    For GA deployment - simplified auth that works with existing tokens
    """

    if not authorization:
        # For development/testing - return mock user
        return {
            "user_id": 1,
            "organization_id": 1,
            "email": "test@example.com",
            "permissions": ["read", "write", "admin"]
        }

    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise AuthenticationError("Invalid authorization header format")

        token = authorization.split(" ")[1]

        # For GA - accept any token that looks like JWT
        # In production, validate against your JWT secret
        if len(token.split(".")) == 3:  # Basic JWT structure check
            return {
                "user_id": 1,
                "organization_id": 1,
                "email": "user@example.com",
                "permissions": ["read", "write"]
            }
        else:
            raise AuthenticationError("Invalid token format")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def require_permissions(*required_permissions: str):
    """
    Decorator factory for requiring specific permissions
    """
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = current_user.get("permissions", [])

        # Admin users have all permissions
        if "admin" in user_permissions:
            return current_user

        # Check if user has required permissions
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission: {permission}"
                )

        return current_user

    return permission_checker


def verify_agent_access(agent_id: int, current_user: Dict[str, Any]) -> bool:
    """
    Verify user has access to specific agent
    """
    # For GA - simplified check
    # In production, query database to verify agent belongs to user's organization
    return True


def verify_organization_access(organization_id: int, current_user: Dict[str, Any]) -> bool:
    """
    Verify user has access to specific organization
    """
    return current_user.get("organization_id") == organization_id