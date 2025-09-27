"""
Base Integration Framework for Third-Party Platforms

This module provides the core abstractions and interfaces for integrating
with external order management systems and product catalogs.
"""

import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
import aiohttp
import hashlib
import base64

from ..core.governance import ConsentContext, governance_engine


class IntegrationType(str, Enum):
    """Types of third-party integrations"""
    ORDER_MANAGEMENT = "order_management"
    PRODUCT_CATALOG = "product_catalog"
    INVENTORY_SYSTEM = "inventory_system"
    CRM_PLATFORM = "crm_platform"
    PAYMENT_PROCESSOR = "payment_processor"
    SHIPPING_PROVIDER = "shipping_provider"


class AuthMethod(str, Enum):
    """Authentication methods for integrations"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    JWT = "jwt"
    WEBHOOK_SECRET = "webhook_secret"


class DataSyncStrategy(str, Enum):
    """Data synchronization strategies"""
    REAL_TIME = "real_time"         # Webhook-based
    POLLING = "polling"             # Regular API calls
    BATCH = "batch"                 # Scheduled batch jobs
    ON_DEMAND = "on_demand"         # Only when requested


@dataclass
class IntegrationCredentials:
    """Secure credential storage for integrations"""
    integration_id: str
    auth_method: AuthMethod
    encrypted_data: str            # Encrypted credential data
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    scopes: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 10
    backoff_factor: float = 1.5


@dataclass
class IntegrationConfig:
    """Configuration for a third-party integration"""
    integration_id: str
    name: str
    integration_type: IntegrationType
    auth_method: AuthMethod

    # Connection details
    base_url: str
    api_version: str = "v1"
    timeout_seconds: int = 30

    # Data sync
    sync_strategy: DataSyncStrategy = DataSyncStrategy.ON_DEMAND
    sync_interval_minutes: int = 60

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Features
    supported_operations: List[str] = field(default_factory=list)
    webhook_endpoints: Dict[str, str] = field(default_factory=dict)

    # Governance
    requires_customer_consent: bool = True
    data_retention_days: int = 30
    pii_fields: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"


@dataclass
class IntegrationResponse:
    """Standardized response from integration operations"""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    response_time_ms: float = 0.0
    rate_limit_remaining: Optional[int] = None

    # Governance
    data_sources: List[str] = field(default_factory=list)
    pii_detected: bool = False
    consent_validated: bool = False


class BaseIntegration(ABC):
    """Base class for all third-party integrations"""

    def __init__(self, config: IntegrationConfig, credentials: IntegrationCredentials):
        self.config = config
        self.credentials = credentials
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(config.rate_limit)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize connection to the third-party service"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def disconnect(self) -> None:
        """Clean up connection resources"""
        if self._session:
            await self._session.close()
            self._session = None

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the third-party service"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the integration is healthy and responsive"""
        pass

    async def make_request(self, method: str, endpoint: str,
                          data: Optional[Dict] = None,
                          params: Optional[Dict] = None,
                          consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Make an authenticated request to the third-party API"""

        start_time = datetime.now()

        # Rate limiting
        await self._rate_limiter.acquire()

        # Build URL
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Prepare headers
        headers = await self._get_auth_headers()

        # Governance check
        if self.config.requires_customer_consent and consent_context:
            from ..core.governance import ConsentScope
            if not governance_engine.validate_consent(consent_context, ConsentScope.USE_CONVERSATION_HISTORY):
                return IntegrationResponse(
                    success=False,
                    error_message="Customer consent required for third-party data access",
                    error_code="CONSENT_REQUIRED"
                )

        try:
            if not self._session:
                await self.connect()

            async with self._session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers
            ) as response:

                response_time_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Update credentials usage
                self.credentials.last_used = datetime.now()
                self.credentials.usage_count += 1

                if response.status == 200:
                    response_data = await response.json()

                    # PII detection
                    pii_detected = self._detect_pii(response_data)

                    # Redact PII if required
                    if pii_detected and consent_context and consent_context.pii_redaction_enabled:
                        response_data = self._redact_pii(response_data)

                    return IntegrationResponse(
                        success=True,
                        data=response_data,
                        response_time_ms=response_time_ms,
                        rate_limit_remaining=self._get_rate_limit_remaining(response),
                        data_sources=[self.config.integration_id],
                        pii_detected=pii_detected,
                        consent_validated=consent_context is not None
                    )

                else:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error_message=f"HTTP {response.status}: {error_text}",
                        error_code=str(response.status),
                        response_time_ms=response_time_ms
                    )

        except asyncio.TimeoutError:
            return IntegrationResponse(
                success=False,
                error_message="Request timeout",
                error_code="TIMEOUT",
                response_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            return IntegrationResponse(
                success=False,
                error_message=str(e),
                error_code="UNKNOWN_ERROR",
                response_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    @abstractmethod
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests"""
        pass

    def _detect_pii(self, data: Any) -> bool:
        """Detect PII in response data"""
        if not self.config.pii_fields:
            return False

        data_str = json.dumps(data) if isinstance(data, (dict, list)) else str(data)

        for pii_field in self.config.pii_fields:
            if pii_field in data_str:
                return True

        # Use governance engine for additional PII detection
        return len(governance_engine.redact_pii(data_str)) != len(data_str)

    def _redact_pii(self, data: Any) -> Any:
        """Redact PII from response data"""
        if isinstance(data, str):
            return governance_engine.redact_pii(data)
        elif isinstance(data, dict):
            return {k: self._redact_pii(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._redact_pii(item) for item in data]
        else:
            return data

    def _get_rate_limit_remaining(self, response: aiohttp.ClientResponse) -> Optional[int]:
        """Extract rate limit information from response headers"""
        # Common header names for rate limiting
        headers_to_check = [
            'X-RateLimit-Remaining',
            'X-Rate-Limit-Remaining',
            'RateLimit-Remaining',
            'X-API-Rate-Limit-Remaining'
        ]

        for header in headers_to_check:
            if header in response.headers:
                try:
                    return int(response.headers[header])
                except ValueError:
                    continue

        return None


class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._tokens = config.burst_allowance
        self._last_refill = datetime.now()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token for making a request"""
        async with self._lock:
            now = datetime.now()

            # Refill tokens based on time elapsed
            time_elapsed = (now - self._last_refill).total_seconds()
            tokens_to_add = time_elapsed * (self.config.requests_per_minute / 60.0)

            self._tokens = min(
                self.config.burst_allowance,
                self._tokens + tokens_to_add
            )
            self._last_refill = now

            # Wait if no tokens available
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / (self.config.requests_per_minute / 60.0)
                await asyncio.sleep(wait_time)
                self._tokens = 1

            self._tokens -= 1


class CredentialManager:
    """Secure credential management for integrations"""

    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key is None:
            encryption_key = base64.b64encode(b"default_key_32_bytes_long_test").decode()
        self._encryption_key = encryption_key
        self._credentials: Dict[str, IntegrationCredentials] = {}

    def _simple_encrypt(self, data: str) -> str:
        """Simple encryption using base64 (for demo purposes)"""
        # In production, use proper encryption like Fernet
        encoded = base64.b64encode(data.encode()).decode()
        return encoded

    def _simple_decrypt(self, encrypted_data: str) -> str:
        """Simple decryption using base64 (for demo purposes)"""
        try:
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            return decoded
        except Exception:
            return ""

    def store_credentials(self, integration_id: str, auth_method: AuthMethod,
                         credential_data: Dict[str, Any]) -> IntegrationCredentials:
        """Store encrypted credentials for an integration"""

        # Encrypt the credential data
        encrypted_data = self._simple_encrypt(json.dumps(credential_data))

        credentials = IntegrationCredentials(
            integration_id=integration_id,
            auth_method=auth_method,
            encrypted_data=encrypted_data
        )

        self._credentials[integration_id] = credentials
        return credentials

    def get_credentials(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt credentials for an integration"""

        if integration_id not in self._credentials:
            return None

        credentials = self._credentials[integration_id]

        try:
            decrypted_data = self._simple_decrypt(credentials.encrypted_data)
            return json.loads(decrypted_data)
        except Exception:
            return None

    def delete_credentials(self, integration_id: str) -> bool:
        """Delete credentials for an integration"""
        return self._credentials.pop(integration_id, None) is not None


# Global credential manager instance
credential_manager = CredentialManager()