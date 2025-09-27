"""
Clean Authentication System
"""

from fastapi import Depends, HTTPException, status, Header
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timezone
import hashlib


JWT_SECRET = "your-secret-key-here"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Simple in-memory user store for testing
# In production, this would be replaced with proper database queries
TEST_USERS = {
    "admin@coconutfurniture.com": {
        "id": 4,
        "email": "admin@coconutfurniture.com",
        "name": "Coconut Admin",
        "password": "password123",
        "is_active": True,
        "is_superuser": False,
        "plan": "professional",
        "organization_id": 4,
        "organization_name": "Coconut Furniture"
    },
    "test@coconutfurniture.com": {
        "id": 2,
        "email": "test@coconutfurniture.com",
        "name": "Coconut Test User",
        "password": "password123",
        "is_active": True,
        "is_superuser": False,
        "plan": "basic",
        "organization_id": 4,
        "organization_name": "Coconut Furniture"
    },
    "demo@aiagents.com": {
        "id": 3,
        "email": "demo@aiagents.com",
        "name": "Demo User",
        "password": "password123",
        "is_active": True,
        "is_superuser": False,
        "plan": "basic",
        "organization_id": 1,
        "organization_name": "Default Organization"
    }
}


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


class SimpleUser:
    """Simple user class for API responses"""
    def __init__(self, id: int, email: str, name: str, is_active: bool = True,
                 is_superuser: bool = False, plan: str = "basic",
                 created_at: datetime = None, updated_at: datetime = None):
        self.id = id
        self.email = email
        self.name = name
        self.is_active = is_active
        self.is_superuser = is_superuser
        self.plan = plan
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at


def get_password_hash(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return get_password_hash(plain_password) == hashed_password


async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with email and password"""
    user_data = TEST_USERS.get(username)
    if not user_data:
        return None

    if not user_data["is_active"]:
        return None

    if user_data["password"] != password:
        return None

    # Return user data without password
    return {k: v for k, v in user_data.items() if k != "password"}


def create_user_token(user_data: Dict[str, Any]) -> str:
    """Create JWT token for user"""
    payload = {
        "user_id": user_data.get("id"),
        "email": user_data.get("email"),
        "organization_id": user_data.get("organization_id"),
        "exp": datetime.now(timezone.utc).timestamp() + (JWT_EXPIRATION_HOURS * 3600)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")


async def get_current_user(authorization: Optional[str] = Header(None)) -> SimpleUser:
    """Get current user from JWT token"""

    # For testing without auth header
    if not authorization:
        # Return default Coconut admin user for testing
        return SimpleUser(
            id=4,
            email="admin@coconutfurniture.com",
            name="Coconut Admin"
        )

    # Extract token from "Bearer <token>" format
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = authorization.split(" ")[1]

    try:
        payload = decode_token(token)
        user_id = payload.get("user_id")
        email = payload.get("email")

        # Find user in our test data
        for user_email, user_data in TEST_USERS.items():
            if user_data["id"] == user_id and user_data["email"] == email:
                return SimpleUser(
                    id=user_data["id"],
                    email=user_data["email"],
                    name=user_data["name"],
                    is_active=user_data["is_active"],
                    is_superuser=user_data["is_superuser"],
                    plan=user_data["plan"]
                )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error"
        )


def get_user_organization_id(user_email: str) -> Optional[int]:
    """Get user's organization ID"""
    user_data = TEST_USERS.get(user_email)
    return user_data.get("organization_id") if user_data else None


def verify_organization_access(organization_id: int, current_user: SimpleUser) -> bool:
    """Verify user has access to specific organization"""
    user_org_id = get_user_organization_id(current_user.email)
    return user_org_id == organization_id


async def require_permissions(*required_permissions: str):
    """Decorator factory for requiring specific permissions (placeholder)"""
    def permission_checker(current_user: SimpleUser = Depends(get_current_user)):
        # For now, all authenticated users have all permissions
        return current_user
    return permission_checker