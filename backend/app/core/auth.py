"""
Secure Authentication System
Fixed critical security vulnerabilities:
- No hardcoded secrets
- Proper bcrypt password hashing
- No cleartext passwords
- Secure authentication flows
"""

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
import logging

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Secure password hashing with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=settings.BCRYPT_ROUNDS)

# OAuth2 scheme for bearer tokens
security_scheme = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


class SimpleUser:
    """Simple user class for API responses"""
    def __init__(self, id: int, email: str, name: str, is_active: bool = True,
                 is_superuser: bool = False, plan: str = "basic",
                 organization_id: Optional[int] = None,
                 created_at: datetime = None, updated_at: datetime = None):
        self.id = id
        self.email = email
        self.name = name
        self.is_active = is_active
        self.is_superuser = is_superuser
        self.plan = plan
        self.organization_id = organization_id
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at


def get_password_hash(password: str) -> str:
    """Securely hash password using bcrypt"""
    if not password:
        raise ValueError("Password cannot be empty")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash"""
    if not plain_password or not hashed_password:
        return False
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.warning(f"Password verification failed: {e}")
        return False


async def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with email and password using database"""
    if not email or not password:
        return None

    try:
        from ..services.database_service import db_service

        # Get user from database
        user = await db_service.get_user_by_email(email)
        if not user:
            logger.info(f"Authentication failed: user not found for email {email}")
            return None

        if not user.is_active:
            logger.info(f"Authentication failed: user inactive for email {email}")
            return None

        # Verify password
        if not verify_password(password, user.password_hash):
            logger.info(f"Authentication failed: invalid password for email {email}")
            return None

        # Return user data without password
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "is_active": user.is_active,
            "is_superuser": getattr(user, "is_superuser", False),
            "plan": getattr(user, "plan", "basic"),
            "organization_id": getattr(user, "organization_id", None)
        }

    except Exception as e:
        logger.error(f"Authentication error for {email}: {e}")
        return None


def create_user_token(user_data: Dict[str, Any]) -> str:
    """Create secure JWT token for user"""
    try:
        payload = {
            "sub": str(user_data.get("id")),
            "email": user_data.get("email"),
            "organization_id": user_data.get("organization_id"),
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
            "type": "access_token"
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise AuthenticationError("Failed to create access token")


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

        # Validate token type
        if payload.get("type") != "access_token":
            raise AuthenticationError("Invalid token type")

        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logger.error(f"Token decode error: {e}")
        raise AuthenticationError("Token validation failed")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> SimpleUser:
    """Get current user from JWT token - SECURE VERSION"""

    # SECURITY FIX: No authentication bypass - always require valid token
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials

    try:
        from ..services.database_service import db_service

        payload = decode_token(token)
        user_id = payload.get("sub")
        email = payload.get("email")

        if not user_id or not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        # Get user from database to ensure they still exist and are active
        user = await db_service.get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive"
            )

        if str(user.id) != user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token user mismatch"
            )

        return SimpleUser(
            id=user.id,
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            is_superuser=getattr(user, "is_superuser", False),
            plan=getattr(user, "plan", "basic"),
            organization_id=getattr(user, "organization_id", None)
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system error"
        )


async def get_user_organization_id(user_email: str) -> Optional[int]:
    """Get user's organization ID from database"""
    try:
        user = await db_service.get_user_by_email(user_email)
        return getattr(user, "organization_id", None) if user else None
    except Exception as e:
        logger.error(f"Error getting user organization: {e}")
        return None


def verify_organization_access(organization_id: int, current_user: SimpleUser) -> bool:
    """Verify user has access to specific organization"""
    if not current_user.organization_id:
        return False
    return current_user.organization_id == organization_id


async def require_admin(current_user: SimpleUser = Depends(get_current_user)) -> SimpleUser:
    """Require admin privileges for endpoint access"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required"
        )
    return current_user


async def require_organization_access(organization_id: int, current_user: SimpleUser = Depends(get_current_user)) -> SimpleUser:
    """Require access to specific organization"""
    if not verify_organization_access(organization_id, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this organization is not permitted"
        )
    return current_user


def create_refresh_token(user_data: Dict[str, Any]) -> str:
    """Create refresh token with longer expiration"""
    try:
        payload = {
            "sub": str(user_data.get("id")),
            "email": user_data.get("email"),
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(days=30),  # 30 days for refresh tokens
            "type": "refresh_token"
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    except Exception as e:
        logger.error(f"Refresh token creation failed: {e}")
        raise AuthenticationError("Failed to create refresh token")


async def refresh_access_token(refresh_token: str) -> str:
    """Create new access token from refresh token"""
    try:
        payload = decode_token(refresh_token)

        if payload.get("type") != "refresh_token":
            raise AuthenticationError("Invalid refresh token")

        email = payload.get("email")
        user = await db_service.get_user_by_email(email)

        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")

        user_data = {
            "id": user.id,
            "email": user.email,
            "organization_id": getattr(user, "organization_id", None)
        }

        return create_user_token(user_data)

    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise AuthenticationError("Failed to refresh token")


# Rate limiting helpers
async def check_rate_limit(identifier: str, max_attempts: int = 5, window_minutes: int = 15) -> bool:
    """Check if rate limit exceeded (implement with Redis in production)"""
    # TODO: Implement with Redis for production
    # For now, always allow (but log the attempt)
    logger.info(f"Rate limit check for {identifier}: {max_attempts} attempts in {window_minutes} minutes")
    return True


async def record_failed_login(email: str, ip_address: str = None):
    """Record failed login attempt for monitoring"""
    logger.warning(f"Failed login attempt for {email} from {ip_address or 'unknown IP'}")
    # TODO: Implement failed login tracking
