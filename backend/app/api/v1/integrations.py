"""
API endpoints for third-party platform integrations

This module provides REST endpoints for managing integrations, authentication,
data access, and governance controls.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field

from ...integrations.base import IntegrationType, AuthMethod, DataSyncStrategy
from ...integrations.auth_manager import auth_manager, PlatformAuthConfigs
from ...integrations.order_management import (
    order_orchestrator, OrderSearchCriteria, OrderStatus
)
from ...integrations.product_catalog import (
    product_recommendation_engine, ProductSearchCriteria, ProductStatus
)
from ...integrations.governance_controls import (
    third_party_governance, ThirdPartyDataType, AccessLevel, ComplianceRequirement
)
from ...integrations.data_mapper import schema_registry
from ...core.governance import ConsentContext, ConsentScope, DataRetentionPolicy

router = APIRouter()


# Request/Response Models
class IntegrationSetupRequest(BaseModel):
    integration_id: str
    name: str
    integration_type: IntegrationType
    auth_method: AuthMethod
    base_url: str
    credentials: Dict[str, Any]
    sync_strategy: DataSyncStrategy = DataSyncStrategy.ON_DEMAND


class ConsentGrantRequest(BaseModel):
    customer_id: str
    platform_id: str
    data_types: List[ThirdPartyDataType]
    access_level: AccessLevel = AccessLevel.READ_ONLY
    purpose_limitation: Optional[List[str]] = None
    expiry_days: int = 365


class OrderSearchRequest(BaseModel):
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    status: Optional[OrderStatus] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 50
    platforms: Optional[List[str]] = None


class ProductSearchRequest(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    status: Optional[ProductStatus] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    in_stock_only: bool = False
    limit: int = 50


class GovernanceValidationRequest(BaseModel):
    customer_id: str
    platform_id: str
    data_types: List[ThirdPartyDataType]
    purpose: str


# Authentication Endpoints
@router.post("/auth/setup/{platform_id}")
async def setup_integration_auth(
    platform_id: str,
    request: IntegrationSetupRequest
):
    """Setup authentication for a third-party integration"""
    try:
        if request.auth_method == AuthMethod.API_KEY:
            success = auth_manager.store_api_key(
                request.integration_id,
                request.credentials.get("api_key"),
                {k: v for k, v in request.credentials.items() if k != "api_key"}
            )

        elif request.auth_method == AuthMethod.BASIC_AUTH:
            success = auth_manager.store_basic_auth(
                request.integration_id,
                request.credentials.get("username"),
                request.credentials.get("password")
            )

        elif request.auth_method == AuthMethod.OAUTH2:
            # For OAuth, we need to return the authorization URL
            if platform_id == "shopify":
                oauth_config = PlatformAuthConfigs.get_shopify_config(
                    request.credentials.get("shop_domain"),
                    request.credentials.get("client_id"),
                    request.credentials.get("client_secret")
                )
            elif platform_id == "bigcommerce":
                oauth_config = PlatformAuthConfigs.get_bigcommerce_config(
                    request.credentials.get("client_id"),
                    request.credentials.get("client_secret")
                )
            else:
                raise HTTPException(status_code=400, detail=f"OAuth not supported for {platform_id}")

            auth_manager.register_oauth_config(request.integration_id, oauth_config)
            auth_url = auth_manager.generate_oauth_url(request.integration_id)

            return {
                "success": True,
                "auth_url": auth_url,
                "message": "Complete OAuth flow using the provided URL"
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported auth method: {request.auth_method}")

        if request.auth_method != AuthMethod.OAUTH2:
            return {
                "success": success,
                "integration_id": request.integration_id,
                "auth_method": request.auth_method.value
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/oauth/callback")
async def oauth_callback(
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="OAuth state parameter")
):
    """Handle OAuth callback from third-party platforms"""
    try:
        result = await auth_manager.handle_oauth_callback(code, state)

        if result.success:
            return {
                "success": True,
                "integration_id": result.data.get("integration_id"),
                "message": "Authentication successful"
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/status/{integration_id}")
async def get_auth_status(integration_id: str):
    """Get authentication status for an integration"""
    try:
        status = auth_manager.get_integration_status(integration_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/auth/{integration_id}")
async def revoke_auth(integration_id: str):
    """Revoke authentication credentials for an integration"""
    try:
        success = auth_manager.revoke_credentials(integration_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Consent Management Endpoints
@router.post("/consent/grant")
async def grant_third_party_consent(request: ConsentGrantRequest):
    """Grant consent for third-party data access"""
    try:
        consent = third_party_governance.grant_third_party_consent(
            customer_id=request.customer_id,
            platform_id=request.platform_id,
            data_types=set(request.data_types),
            access_level=request.access_level,
            purpose_limitation=request.purpose_limitation,
            expiry_days=request.expiry_days
        )

        return {
            "success": True,
            "consent_id": f"{request.customer_id}_{request.platform_id}",
            "expires_at": consent.expiry_date.isoformat() if consent.expiry_date else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/consent/{customer_id}/{platform_id}")
async def revoke_third_party_consent(customer_id: str, platform_id: str):
    """Revoke third-party data access consent"""
    try:
        success = third_party_governance.revoke_third_party_consent(customer_id, platform_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consent/{customer_id}")
async def get_customer_consents(customer_id: str):
    """Get all third-party consents for a customer"""
    try:
        consents = third_party_governance.get_customer_third_party_consents(customer_id)

        return {
            "customer_id": customer_id,
            "consents": [
                {
                    "platform_id": consent.platform_id,
                    "allowed_data_types": [dt.value for dt in consent.allowed_data_types],
                    "access_level": consent.access_level.value,
                    "expires_at": consent.expiry_date.isoformat() if consent.expiry_date else None,
                    "last_used": consent.last_used.isoformat() if consent.last_used else None
                }
                for consent in consents
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Order Management Endpoints
@router.post("/orders/search")
async def search_orders(request: OrderSearchRequest):
    """Search orders across integrated platforms"""
    try:
        # Create search criteria
        criteria = OrderSearchCriteria(
            customer_id=request.customer_id,
            customer_email=request.customer_email,
            status=request.status,
            date_from=request.date_from,
            date_to=request.date_to,
            limit=request.limit
        )

        # Basic consent context (would be properly constructed in real app)
        consent_context = ConsentContext(
            consents={ConsentScope.USE_CONVERSATION_HISTORY, ConsentScope.PERSONALIZE_RESPONSES}
        )

        # Search across platforms
        results = await order_orchestrator.search_orders_across_platforms(
            criteria, request.platforms, consent_context
        )

        # Format response
        response = {
            "search_criteria": {
                "customer_id": request.customer_id,
                "customer_email": request.customer_email,
                "status": request.status.value if request.status else None,
                "platforms_searched": list(results.keys())
            },
            "results": {}
        }

        for platform, result in results.items():
            if result.success:
                response["results"][platform] = {
                    "success": True,
                    "orders": result.data.get("orders", []),
                    "count": len(result.data.get("orders", []))
                }
            else:
                response["results"][platform] = {
                    "success": False,
                    "error": result.error_message
                }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders/insights/{customer_email}")
async def get_order_insights(customer_email: str):
    """Get customer insights from order history"""
    try:
        consent_context = ConsentContext(
            consents={ConsentScope.USE_CONVERSATION_HISTORY, ConsentScope.ANALYZE_BEHAVIOR}
        )

        insights = await order_orchestrator.get_order_insights(customer_email, consent_context)

        return {
            "customer_email": customer_email,
            "insights": insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Product Catalog Endpoints
@router.post("/products/search")
async def search_products(request: ProductSearchRequest):
    """Search products across integrated catalogs"""
    try:
        criteria = ProductSearchCriteria(
            query=request.query,
            category=request.category,
            status=request.status,
            price_min=request.price_min,
            price_max=request.price_max,
            in_stock_only=request.in_stock_only,
            limit=request.limit
        )

        consent_context = ConsentContext(
            consents={ConsentScope.USE_CONVERSATION_HISTORY, ConsentScope.PERSONALIZE_RESPONSES}
        )

        results = await product_recommendation_engine.search_products_intelligent(
            request.query or "",
            customer_context={},
            consent_context=consent_context
        )

        return {
            "search_query": request.query,
            "results": {
                platform: [
                    {
                        "product_id": product.product_id,
                        "title": product.title,
                        "price": product.price,
                        "currency": product.currency,
                        "status": product.status.value,
                        "data_source": product.data_source
                    }
                    for product in products
                ]
                for platform, products in results.items()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/recommendations/{customer_email}")
async def get_personalized_recommendations(customer_email: str):
    """Get personalized product recommendations"""
    try:
        consent_context = ConsentContext(
            consents={ConsentScope.PERSONALIZE_RESPONSES, ConsentScope.ANALYZE_BEHAVIOR}
        )

        recommendations = await product_recommendation_engine.get_personalized_recommendations(
            customer_email,
            context={},
            consent_context=consent_context
        )

        return {
            "customer_email": customer_email,
            "recommendations": [
                {
                    "product_id": product.product_id,
                    "title": product.title,
                    "price": product.price,
                    "currency": product.currency,
                    "data_source": product.data_source,
                    "categories": product.categories
                }
                for product in recommendations
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Data Transformation Endpoints
@router.get("/schemas")
async def list_schemas():
    """List all available data transformation schemas"""
    try:
        schemas = schema_registry.list_schemas()
        return {"schemas": schemas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transform/{platform_id}")
async def transform_data(
    platform_id: str,
    source_data: Dict[str, Any] = Body(...),
    customer_id: Optional[str] = None
):
    """Transform data using platform schema"""
    try:
        consent_context = ConsentContext(
            consents={ConsentScope.USE_CONVERSATION_HISTORY},
            pii_redaction_enabled=True
        ) if customer_id else None

        mapper = schema_registry.get_mapper()
        result = await mapper.transform_data(
            platform_id, source_data, consent_context=consent_context
        )

        return {
            "success": result.success,
            "transformed_data": result.transformed_data,
            "fields_processed": result.fields_processed,
            "fields_skipped": result.fields_skipped,
            "pii_fields_redacted": result.pii_fields_redacted,
            "errors": result.errors,
            "warnings": result.warnings
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Governance and Compliance Endpoints
@router.get("/governance/compliance/{integration_id}")
async def get_compliance_status(integration_id: str):
    """Get compliance status for an integration"""
    try:
        status = third_party_governance.get_integration_compliance_status(integration_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance/audit")
async def get_audit_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...)
):
    """Generate compliance audit report"""
    try:
        report = third_party_governance.generate_compliance_report(start_date, end_date)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/validate-access")
async def validate_data_access(request: GovernanceValidationRequest):
    """Validate if third-party data access is permitted"""
    try:
        consent_context = ConsentContext(
            consents={ConsentScope.USE_CONVERSATION_HISTORY, ConsentScope.PERSONALIZE_RESPONSES}
        )

        result = third_party_governance.validate_third_party_access(
            request.customer_id, request.platform_id, request.data_types, request.purpose, consent_context
        )

        return {
            "access_permitted": result.success,
            "error_message": result.error_message,
            "error_code": result.error_code
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health Check Endpoint
@router.get("/health")
async def integration_health_check():
    """Check health of all integrations"""
    try:
        # This would check all registered integrations
        # For now, return a simple status
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "integrations": {
                "auth_manager": "operational",
                "order_orchestrator": "operational",
                "product_engine": "operational",
                "governance_engine": "operational"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))