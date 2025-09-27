"""
Authentication and API Key Management for Third-Party Integrations

This module provides secure credential storage, OAuth flows, and API key management
for all third-party platform integrations.
"""

import json
import asyncio
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import aiohttp
from urllib.parse import urlencode, parse_qs, urlparse

from .base import (
    IntegrationConfig, IntegrationCredentials, AuthMethod,
    credential_manager, IntegrationResponse
)
from ..core.governance import ConsentContext, governance_engine


class TokenStatus(str, Enum):
    """OAuth token status"""
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    REFRESH_NEEDED = "refresh_needed"


@dataclass
class OAuthConfig:
    """OAuth configuration for a platform"""
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    scope: List[str] = field(default_factory=list)
    redirect_uri: Optional[str] = None


@dataclass
class OAuthToken:
    """OAuth token information"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

    # Computed fields
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def expires_at(self) -> Optional[datetime]:
        if self.expires_in:
            return self.created_at + timedelta(seconds=self.expires_in)
        return None

    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    @property
    def needs_refresh(self) -> bool:
        if self.expires_at:
            # Refresh if expires within 5 minutes
            return datetime.now() > (self.expires_at - timedelta(minutes=5))
        return False


@dataclass
class WebhookConfig:
    """Webhook configuration for real-time data sync"""
    webhook_id: str
    integration_id: str
    webhook_url: str
    events: List[str]
    secret: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class AuthenticationManager:
    """Manages authentication for all third-party integrations"""

    def __init__(self):
        self.oauth_configs: Dict[str, OAuthConfig] = {}
        self.oauth_tokens: Dict[str, OAuthToken] = {}
        self.webhook_configs: Dict[str, WebhookConfig] = {}
        self.pending_oauth_states: Dict[str, Dict[str, Any]] = {}

    def register_oauth_config(self, integration_id: str, config: OAuthConfig):
        """Register OAuth configuration for an integration"""
        self.oauth_configs[integration_id] = config

    def generate_oauth_url(self, integration_id: str, state: Optional[str] = None) -> str:
        """Generate OAuth authorization URL"""
        if integration_id not in self.oauth_configs:
            raise ValueError(f"OAuth config not found for {integration_id}")

        config = self.oauth_configs[integration_id]

        if not state:
            state = secrets.token_urlsafe(32)

        # Store state for verification
        self.pending_oauth_states[state] = {
            "integration_id": integration_id,
            "created_at": datetime.now()
        }

        params = {
            "client_id": config.client_id,
            "response_type": "code",
            "scope": " ".join(config.scope),
            "state": state
        }

        if config.redirect_uri:
            params["redirect_uri"] = config.redirect_uri

        return f"{config.authorization_url}?{urlencode(params)}"

    async def handle_oauth_callback(self, authorization_code: str, state: str) -> IntegrationResponse:
        """Handle OAuth callback and exchange code for tokens"""

        # Verify state
        if state not in self.pending_oauth_states:
            return IntegrationResponse(
                success=False,
                error_message="Invalid OAuth state",
                error_code="INVALID_STATE"
            )

        state_info = self.pending_oauth_states.pop(state)
        integration_id = state_info["integration_id"]

        # Check state expiry (30 minutes)
        if datetime.now() - state_info["created_at"] > timedelta(minutes=30):
            return IntegrationResponse(
                success=False,
                error_message="OAuth state expired",
                error_code="STATE_EXPIRED"
            )

        config = self.oauth_configs[integration_id]

        # Exchange code for tokens
        token_data = {
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "code": authorization_code
        }

        if config.redirect_uri:
            token_data["redirect_uri"] = config.redirect_uri

        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.token_url,
                data=token_data,
                headers={"Accept": "application/json"}
            ) as response:

                if response.status == 200:
                    token_response = await response.json()

                    # Create OAuth token
                    oauth_token = OAuthToken(
                        access_token=token_response["access_token"],
                        token_type=token_response.get("token_type", "Bearer"),
                        expires_in=token_response.get("expires_in"),
                        refresh_token=token_response.get("refresh_token"),
                        scope=token_response.get("scope")
                    )

                    # Store token
                    self.oauth_tokens[integration_id] = oauth_token

                    # Store encrypted credentials
                    credential_data = {
                        "access_token": oauth_token.access_token,
                        "refresh_token": oauth_token.refresh_token,
                        "token_type": oauth_token.token_type,
                        "expires_at": oauth_token.expires_at.isoformat() if oauth_token.expires_at else None
                    }

                    credential_manager.store_credentials(
                        integration_id,
                        AuthMethod.OAUTH2,
                        credential_data
                    )

                    return IntegrationResponse(
                        success=True,
                        data={"integration_id": integration_id, "token_type": oauth_token.token_type}
                    )

                else:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error_message=f"Token exchange failed: {error_text}",
                        error_code="TOKEN_EXCHANGE_FAILED"
                    )

    async def refresh_oauth_token(self, integration_id: str) -> IntegrationResponse:
        """Refresh an expired OAuth token"""

        if integration_id not in self.oauth_tokens:
            return IntegrationResponse(
                success=False,
                error_message="No OAuth token found",
                error_code="NO_TOKEN"
            )

        token = self.oauth_tokens[integration_id]
        if not token.refresh_token:
            return IntegrationResponse(
                success=False,
                error_message="No refresh token available",
                error_code="NO_REFRESH_TOKEN"
            )

        config = self.oauth_configs[integration_id]

        refresh_data = {
            "grant_type": "refresh_token",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "refresh_token": token.refresh_token
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.token_url,
                data=refresh_data,
                headers={"Accept": "application/json"}
            ) as response:

                if response.status == 200:
                    token_response = await response.json()

                    # Update token
                    token.access_token = token_response["access_token"]
                    token.expires_in = token_response.get("expires_in")
                    token.created_at = datetime.now()

                    if "refresh_token" in token_response:
                        token.refresh_token = token_response["refresh_token"]

                    # Update stored credentials
                    credential_data = {
                        "access_token": token.access_token,
                        "refresh_token": token.refresh_token,
                        "token_type": token.token_type,
                        "expires_at": token.expires_at.isoformat() if token.expires_at else None
                    }

                    credential_manager.store_credentials(
                        integration_id,
                        AuthMethod.OAUTH2,
                        credential_data
                    )

                    return IntegrationResponse(
                        success=True,
                        data={"access_token": token.access_token}
                    )

                else:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error_message=f"Token refresh failed: {error_text}",
                        error_code="TOKEN_REFRESH_FAILED"
                    )

    def store_api_key(self, integration_id: str, api_key: str, additional_data: Dict[str, Any] = None) -> bool:
        """Store API key credentials"""
        credential_data = {"api_key": api_key}
        if additional_data:
            credential_data.update(additional_data)

        credentials = credential_manager.store_credentials(
            integration_id,
            AuthMethod.API_KEY,
            credential_data
        )

        return credentials is not None

    def store_basic_auth(self, integration_id: str, username: str, password: str) -> bool:
        """Store basic authentication credentials"""
        credential_data = {
            "username": username,
            "password": password
        }

        credentials = credential_manager.store_credentials(
            integration_id,
            AuthMethod.BASIC_AUTH,
            credential_data
        )

        return credentials is not None

    async def validate_credentials(self, integration_id: str) -> TokenStatus:
        """Validate stored credentials for an integration"""

        credentials = credential_manager.get_credentials(integration_id)
        if not credentials:
            return TokenStatus.REVOKED

        if integration_id in self.oauth_tokens:
            token = self.oauth_tokens[integration_id]

            if token.is_expired:
                if token.refresh_token:
                    return TokenStatus.REFRESH_NEEDED
                else:
                    return TokenStatus.EXPIRED

            return TokenStatus.VALID

        # For API keys and basic auth, assume valid unless proven otherwise
        return TokenStatus.VALID

    async def get_valid_token(self, integration_id: str) -> Optional[str]:
        """Get a valid access token, refreshing if necessary"""

        status = await self.validate_credentials(integration_id)

        if status == TokenStatus.REFRESH_NEEDED:
            refresh_result = await self.refresh_oauth_token(integration_id)
            if not refresh_result.success:
                return None

        if integration_id in self.oauth_tokens:
            return self.oauth_tokens[integration_id].access_token

        # For API keys
        credentials = credential_manager.get_credentials(integration_id)
        if credentials and "api_key" in credentials:
            return credentials["api_key"]

        return None

    def register_webhook(self, integration_id: str, webhook_url: str,
                        events: List[str], secret: Optional[str] = None) -> WebhookConfig:
        """Register a webhook for real-time data updates"""

        if not secret:
            secret = secrets.token_urlsafe(32)

        webhook_id = f"{integration_id}_{secrets.token_urlsafe(8)}"

        webhook_config = WebhookConfig(
            webhook_id=webhook_id,
            integration_id=integration_id,
            webhook_url=webhook_url,
            events=events,
            secret=secret
        )

        self.webhook_configs[webhook_id] = webhook_config

        return webhook_config

    def verify_webhook_signature(self, webhook_id: str, payload: bytes,
                                signature: str, algorithm: str = "sha256") -> bool:
        """Verify webhook signature for security"""

        if webhook_id not in self.webhook_configs:
            return False

        webhook_config = self.webhook_configs[webhook_id]

        import hmac
        import hashlib

        expected_signature = hmac.new(
            webhook_config.secret.encode(),
            payload,
            getattr(hashlib, algorithm)
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def revoke_credentials(self, integration_id: str) -> bool:
        """Revoke and delete credentials for an integration"""

        # Remove from memory
        self.oauth_tokens.pop(integration_id, None)

        # Remove from secure storage
        return credential_manager.delete_credentials(integration_id)

    def get_integration_status(self, integration_id: str) -> Dict[str, Any]:
        """Get authentication status for an integration"""

        credentials = credential_manager.get_credentials(integration_id)

        if not credentials:
            return {
                "authenticated": False,
                "auth_method": None,
                "status": "no_credentials"
            }

        if integration_id in self.oauth_tokens:
            token = self.oauth_tokens[integration_id]
            return {
                "authenticated": True,
                "auth_method": "oauth2",
                "status": "valid" if not token.is_expired else "expired",
                "expires_at": token.expires_at.isoformat() if token.expires_at else None,
                "has_refresh_token": token.refresh_token is not None
            }

        return {
            "authenticated": True,
            "auth_method": "api_key",
            "status": "valid"
        }


class PlatformAuthConfigs:
    """Pre-configured OAuth settings for popular platforms"""

    SHOPIFY = {
        "authorization_url": "https://{shop_domain}/admin/oauth/authorize",
        "token_url": "https://{shop_domain}/admin/oauth/access_token",
        "scope": ["read_orders", "read_products", "read_customers"]
    }

    WOOCOMMERCE = {
        # WooCommerce uses custom OAuth implementation
        "authorization_url": "{site_url}/wc-auth/v1/authorize",
        "token_url": "{site_url}/wc-auth/v1/access_token",
        "scope": ["read", "write"]
    }

    BIGCOMMERCE = {
        "authorization_url": "https://login.bigcommerce.com/oauth2/authorize",
        "token_url": "https://login.bigcommerce.com/oauth2/token",
        "scope": ["store_v2_orders", "store_v2_products"]
    }

    @classmethod
    def get_shopify_config(cls, shop_domain: str, client_id: str, client_secret: str) -> OAuthConfig:
        """Get Shopify OAuth configuration"""
        return OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            authorization_url=cls.SHOPIFY["authorization_url"].format(shop_domain=shop_domain),
            token_url=cls.SHOPIFY["token_url"].format(shop_domain=shop_domain),
            scope=cls.SHOPIFY["scope"]
        )

    @classmethod
    def get_bigcommerce_config(cls, client_id: str, client_secret: str) -> OAuthConfig:
        """Get BigCommerce OAuth configuration"""
        return OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            authorization_url=cls.BIGCOMMERCE["authorization_url"],
            token_url=cls.BIGCOMMERCE["token_url"],
            scope=cls.BIGCOMMERCE["scope"]
        )


# Global authentication manager instance
auth_manager = AuthenticationManager()