"""
Authentication utilities for JWT tokens and password hashing.
"""

from typing import Optional, Union
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_db
from ..services.database_service import db_service

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

# JWT settings
ALGORITHM = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Extract token from credentials
        token = credentials.credentials
        payload = verify_token(token)

        if payload is None:
            raise credentials_exception

        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception

    except (JWTError, ValueError):
        raise credentials_exception

    # Get user from database
    user = await db_service.get_user_by_id(int(user_id))
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    return user


async def get_current_active_user(current_user = Depends(get_current_user)):
    """Get current active user (convenience function)"""
    return current_user


def create_user_token(user_id: int, email: str) -> str:
    """Create access token for user"""
    token_data = {
        "sub": str(user_id),
        "email": email,
        "type": "access"
    }
    return create_access_token(token_data)


async def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate user with email and password"""
    user = await db_service.get_user_by_email(email)
    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


class AuthError(Exception):
    """Custom authentication error"""
    pass


def require_plan(required_plan: str):
    """Decorator to require specific user plan"""
    def plan_checker(current_user = Depends(get_current_user)):
        plan_hierarchy = {
            "free": 0,
            "pro": 1,
            "enterprise": 2
        }

        user_plan_level = plan_hierarchy.get(current_user.plan, 0)
        required_plan_level = plan_hierarchy.get(required_plan, 0)

        if user_plan_level < required_plan_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {required_plan} plan or higher"
            )

        return current_user

    return plan_checker


def optional_auth():
    """Optional authentication - returns user if authenticated, None otherwise"""
    async def _optional_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        db: AsyncSession = Depends(get_db)
    ):
        if not credentials:
            return None

        try:
            token = credentials.credentials
            payload = verify_token(token)

            if payload is None:
                return None

            user_id: int = payload.get("sub")
            if user_id is None:
                return None

            user = await db_service.get_user_by_id(int(user_id))
            if user and user.is_active:
                return user

        except (JWTError, ValueError):
            pass

        return None

    return _optional_auth