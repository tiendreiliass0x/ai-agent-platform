"""
Integration Orchestrator Service

This module provides the main orchestrator that coordinates all third-party
integrations, manages data flow, and ensures governance compliance.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from .base import IntegrationConfig, IntegrationCredentials, IntegrationType, AuthMethod
from .auth_manager import auth_manager, PlatformAuthConfigs
from .order_management import (
    order_orchestrator, ShopifyIntegration, WooCommerceIntegration, BigCommerceIntegration,
    OrderSearchCriteria, StandardizedOrder
)
from .product_catalog import (
    product_recommendation_engine, ShopifyProductCatalog, WooCommerceProductCatalog,
    ProductSearchCriteria, StandardizedProduct
)
from .governance_controls import third_party_governance, ThirdPartyDataType, AccessLevel
from .data_mapper import schema_registry
from ..core.governance import ConsentContext, ConsentScope


class IntegrationStatus(str, Enum):
    """Status of an integration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    AUTHENTICATING = "authenticating"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class IntegrationInstance:
    """Represents a configured integration instance"""
    integration_id: str
    config: IntegrationConfig
    credentials: IntegrationCredentials
    integration_instance: Any  # The actual integration object
    status: IntegrationStatus = IntegrationStatus.INACTIVE
    last_sync: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomerContext:
    """Customer context for integration operations"""
    customer_id: str
    customer_email: str
    consent_context: ConsentContext
    preferences: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None


class IntegrationOrchestrator:
    """Main orchestrator for all third-party integrations"""

    def __init__(self):
        self.integrations: Dict[str, IntegrationInstance] = {}
        self.integration_registry = {
            # Order Management
            ("shopify", IntegrationType.ORDER_MANAGEMENT): ShopifyIntegration,
            ("woocommerce", IntegrationType.ORDER_MANAGEMENT): WooCommerceIntegration,
            ("bigcommerce", IntegrationType.ORDER_MANAGEMENT): BigCommerceIntegration,

            # Product Catalog
            ("shopify", IntegrationType.PRODUCT_CATALOG): ShopifyProductCatalog,
            ("woocommerce", IntegrationType.PRODUCT_CATALOG): WooCommerceProductCatalog,
        }

    async def register_integration(self, platform_id: str, integration_type: IntegrationType,
                                 config: IntegrationConfig,
                                 credentials: IntegrationCredentials) -> bool:
        """Register a new integration instance"""

        integration_key = (platform_id, integration_type)
        if integration_key not in self.integration_registry:
            raise ValueError(f"Unsupported integration: {platform_id} {integration_type}")

        integration_class = self.integration_registry[integration_key]

        try:
            # Create integration instance
            integration_instance = integration_class(config, credentials)

            # Test authentication
            auth_success = await integration_instance.authenticate()
            if not auth_success:
                return False

            # Store integration
            instance = IntegrationInstance(
                integration_id=config.integration_id,
                config=config,
                credentials=credentials,
                integration_instance=integration_instance,
                status=IntegrationStatus.ACTIVE
            )

            self.integrations[config.integration_id] = instance

            # Register with specialized orchestrators
            if integration_type == IntegrationType.ORDER_MANAGEMENT:
                order_orchestrator.register_integration(config.integration_id, integration_instance)

            elif integration_type == IntegrationType.PRODUCT_CATALOG:
                product_recommendation_engine.register_integration(config.integration_id, integration_instance)

            return True

        except Exception as e:
            # Store failed integration for debugging
            instance = IntegrationInstance(
                integration_id=config.integration_id,
                config=config,
                credentials=credentials,
                integration_instance=None,
                status=IntegrationStatus.ERROR,
                error_message=str(e)
            )
            self.integrations[config.integration_id] = instance
            return False

    async def get_customer_unified_profile(self, customer_context: CustomerContext) -> Dict[str, Any]:
        """Get unified customer profile across all integrations"""

        # Validate consent
        validation_result = third_party_governance.validate_third_party_access(
            customer_context.customer_id,
            "unified_profile",
            [ThirdPartyDataType.ORDER_HISTORY, ThirdPartyDataType.CUSTOMER_PROFILE],
            "customer_support",
            customer_context.consent_context
        )

        if not validation_result.success:
            return {"error": validation_result.error_message}

        profile = {
            "customer_id": customer_context.customer_id,
            "customer_email": customer_context.customer_email,
            "profile_generated_at": datetime.now().isoformat(),
            "data_sources": [],
            "order_history": {},
            "product_interactions": {},
            "insights": {}
        }

        # Gather order history
        try:
            order_insights = await order_orchestrator.get_order_insights(
                customer_context.customer_email,
                customer_context.consent_context
            )
            profile["order_history"] = order_insights
            profile["data_sources"].append("order_management")
        except Exception as e:
            profile["order_history"] = {"error": str(e)}

        # Get product recommendations
        try:
            recommendations = await product_recommendation_engine.get_personalized_recommendations(
                customer_context.customer_email,
                context=customer_context.preferences,
                consent_context=customer_context.consent_context
            )

            profile["product_interactions"] = {
                "personalized_recommendations": [
                    {
                        "product_id": product.product_id,
                        "title": product.title,
                        "price": product.price,
                        "data_source": product.data_source
                    }
                    for product in recommendations[:5]
                ]
            }
            profile["data_sources"].append("product_catalog")
        except Exception as e:
            profile["product_interactions"] = {"error": str(e)}

        # Generate insights
        profile["insights"] = await self._generate_customer_insights(customer_context, profile)

        # Create audit record
        third_party_governance.create_access_audit(
            customer_id=customer_context.customer_id,
            platform_id="unified_profile",
            integration_id="orchestrator",
            data_types=[ThirdPartyDataType.ORDER_HISTORY, ThirdPartyDataType.CUSTOMER_PROFILE],
            purpose="unified_customer_profile",
            access_level=AccessLevel.READ_ONLY,
            records_accessed=1,
            consent_validated=True,
            pii_redacted=customer_context.consent_context.pii_redaction_enabled,
            session_id=customer_context.session_id
        )

        return profile

    async def _generate_customer_insights(self, customer_context: CustomerContext,
                                        profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI insights from customer data"""

        insights = {
            "customer_tier": "standard",
            "engagement_level": "medium",
            "preferences": [],
            "recommendations": [],
            "risk_factors": []
        }

        order_history = profile.get("order_history", {})

        # Customer tier analysis
        total_spent = order_history.get("total_spent", 0)
        total_orders = order_history.get("total_orders", 0)

        if total_spent > 1000 or total_orders > 10:
            insights["customer_tier"] = "premium"
        elif total_spent > 500 or total_orders > 5:
            insights["customer_tier"] = "valued"

        # Engagement analysis
        recent_orders = order_history.get("recent_orders_90d", 0)
        if recent_orders >= 3:
            insights["engagement_level"] = "high"
        elif recent_orders >= 1:
            insights["engagement_level"] = "medium"
        else:
            insights["engagement_level"] = "low"

        # Preference inference
        platform_insights = order_history.get("insights", [])
        insights["preferences"] = platform_insights

        # Risk assessment
        if "Has returned items previously" in platform_insights:
            insights["risk_factors"].append("return_risk")

        # Recommendations
        if insights["customer_tier"] == "premium":
            insights["recommendations"].append("offer_vip_support")

        if insights["engagement_level"] == "low":
            insights["recommendations"].append("re_engagement_campaign")

        return insights

    async def search_across_platforms(self, search_type: str, criteria: Dict[str, Any],
                                    customer_context: CustomerContext) -> Dict[str, Any]:
        """Search across all integrated platforms"""

        if search_type == "orders":
            # Convert to OrderSearchCriteria
            order_criteria = OrderSearchCriteria(
                customer_email=criteria.get("customer_email"),
                customer_id=criteria.get("customer_id"),
                status=criteria.get("status"),
                date_from=criteria.get("date_from"),
                date_to=criteria.get("date_to"),
                limit=criteria.get("limit", 50)
            )

            results = await order_orchestrator.search_orders_across_platforms(
                order_criteria,
                platforms=criteria.get("platforms"),
                consent_context=customer_context.consent_context
            )

            return {"search_type": "orders", "results": results}

        elif search_type == "products":
            # Convert to ProductSearchCriteria
            product_criteria = ProductSearchCriteria(
                query=criteria.get("query"),
                category=criteria.get("category"),
                status=criteria.get("status"),
                price_min=criteria.get("price_min"),
                price_max=criteria.get("price_max"),
                in_stock_only=criteria.get("in_stock_only", False),
                limit=criteria.get("limit", 50)
            )

            results = await product_recommendation_engine.search_products_intelligent(
                criteria.get("query", ""),
                customer_context=customer_context.preferences,
                consent_context=customer_context.consent_context
            )

            return {"search_type": "products", "results": results}

        else:
            raise ValueError(f"Unsupported search type: {search_type}")

    async def sync_data_from_platform(self, integration_id: str,
                                    customer_context: Optional[CustomerContext] = None) -> Dict[str, Any]:
        """Sync data from a specific platform"""

        if integration_id not in self.integrations:
            return {"success": False, "error": "Integration not found"}

        instance = self.integrations[integration_id]

        try:
            # Update status
            instance.status = IntegrationStatus.ACTIVE
            instance.last_sync = datetime.now()

            # Test health
            health_ok = await instance.integration_instance.health_check()
            if not health_ok:
                instance.status = IntegrationStatus.ERROR
                return {"success": False, "error": "Health check failed"}

            # Sync based on integration type
            if instance.config.integration_type == IntegrationType.ORDER_MANAGEMENT:
                # Sync recent orders
                if customer_context:
                    criteria = OrderSearchCriteria(
                        customer_email=customer_context.customer_email,
                        date_from=datetime.now() - timedelta(days=30),
                        limit=100
                    )
                    result = await instance.integration_instance.get_orders(
                        criteria, customer_context.consent_context
                    )
                    sync_result = {"orders_synced": len(result.data.get("orders", [])) if result.success else 0}
                else:
                    sync_result = {"message": "No customer context provided"}

            elif instance.config.integration_type == IntegrationType.PRODUCT_CATALOG:
                # Sync product catalog
                criteria = ProductSearchCriteria(limit=100)
                result = await instance.integration_instance.search_products(
                    criteria, customer_context.consent_context if customer_context else None
                )
                sync_result = {"products_synced": len(result.data.get("products", [])) if result.success else 0}

            else:
                sync_result = {"message": "Sync not implemented for this integration type"}

            # Update metrics
            instance.metrics["last_sync_result"] = sync_result
            instance.metrics["sync_count"] = instance.metrics.get("sync_count", 0) + 1

            return {"success": True, "sync_result": sync_result}

        except Exception as e:
            instance.status = IntegrationStatus.ERROR
            instance.error_message = str(e)
            return {"success": False, "error": str(e)}

    async def health_check_all_integrations(self) -> Dict[str, Any]:
        """Perform health check on all integrations"""

        health_results = {}

        for integration_id, instance in self.integrations.items():
            try:
                if instance.integration_instance:
                    health_ok = await instance.integration_instance.health_check()
                    instance.status = IntegrationStatus.ACTIVE if health_ok else IntegrationStatus.ERROR

                    health_results[integration_id] = {
                        "status": instance.status.value,
                        "platform": instance.config.base_url,
                        "integration_type": instance.config.integration_type.value,
                        "last_sync": instance.last_sync.isoformat() if instance.last_sync else None,
                        "error_message": instance.error_message
                    }
                else:
                    health_results[integration_id] = {
                        "status": "not_initialized",
                        "error_message": instance.error_message
                    }

            except Exception as e:
                health_results[integration_id] = {
                    "status": "error",
                    "error_message": str(e)
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_integrations": len(self.integrations),
            "healthy_integrations": len([r for r in health_results.values() if r["status"] == "active"]),
            "results": health_results
        }

    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get metrics for all integrations"""

        metrics = {
            "total_integrations": len(self.integrations),
            "by_status": {},
            "by_type": {},
            "by_platform": {},
            "recent_activity": []
        }

        # Count by status
        for instance in self.integrations.values():
            status = instance.status.value
            metrics["by_status"][status] = metrics["by_status"].get(status, 0) + 1

            # Count by type
            int_type = instance.config.integration_type.value
            metrics["by_type"][int_type] = metrics["by_type"].get(int_type, 0) + 1

            # Count by platform (extract from base_url)
            platform = instance.config.base_url.split('/')[2].split('.')[0]  # Simple extraction
            metrics["by_platform"][platform] = metrics["by_platform"].get(platform, 0) + 1

            # Recent activity
            if instance.last_sync and (datetime.now() - instance.last_sync).days <= 1:
                metrics["recent_activity"].append({
                    "integration_id": instance.integration_id,
                    "last_sync": instance.last_sync.isoformat(),
                    "sync_count": instance.metrics.get("sync_count", 0)
                })

        return metrics

    async def remove_integration(self, integration_id: str) -> bool:
        """Remove an integration"""

        if integration_id not in self.integrations:
            return False

        instance = self.integrations[integration_id]

        # Clean up from specialized orchestrators
        try:
            if instance.config.integration_type == IntegrationType.ORDER_MANAGEMENT:
                # Would need to add removal method to order_orchestrator
                pass

            elif instance.config.integration_type == IntegrationType.PRODUCT_CATALOG:
                # Would need to add removal method to product_recommendation_engine
                pass

        except Exception:
            pass  # Best effort cleanup

        # Remove from auth manager
        auth_manager.revoke_credentials(integration_id)

        # Remove from orchestrator
        del self.integrations[integration_id]

        return True


# Global integration orchestrator instance
integration_orchestrator = IntegrationOrchestrator()