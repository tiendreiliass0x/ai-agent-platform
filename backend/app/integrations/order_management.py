"""
Order Management Platform Integrations

This module provides connectors for popular order management systems
including Shopify, WooCommerce, BigCommerce, and custom APIs.
"""

import json
import hmac
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from .base import (
    BaseIntegration, IntegrationConfig, IntegrationCredentials, IntegrationResponse,
    IntegrationType, AuthMethod, DataSyncStrategy, credential_manager
)
from ..core.governance import ConsentContext


class OrderStatus(str, Enum):
    """Standardized order status across platforms"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    RETURNED = "returned"


@dataclass
class StandardizedOrder:
    """Standardized order representation across platforms"""
    order_id: str
    platform_order_id: str
    customer_id: str
    customer_email: str

    # Order details
    status: OrderStatus
    total_amount: float
    currency: str
    created_at: datetime
    updated_at: datetime

    # Items
    line_items: List[Dict[str, Any]] = field(default_factory=list)

    # Addresses
    billing_address: Optional[Dict[str, str]] = None
    shipping_address: Optional[Dict[str, str]] = None

    # Fulfillment
    tracking_numbers: List[str] = field(default_factory=list)
    shipping_method: Optional[str] = None
    estimated_delivery: Optional[datetime] = None

    # Platform-specific data
    platform_data: Dict[str, Any] = field(default_factory=dict)

    # Governance
    data_source: str = ""
    consent_validated: bool = False


@dataclass
class OrderSearchCriteria:
    """Search criteria for order queries"""
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    order_ids: Optional[List[str]] = None
    status: Optional[OrderStatus] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 50
    offset: int = 0


class ShopifyIntegration(BaseIntegration):
    """Shopify order management integration"""

    def __init__(self, config: IntegrationConfig, credentials: IntegrationCredentials):
        super().__init__(config, credentials)
        self.shop_domain = None

    async def authenticate(self) -> bool:
        """Authenticate with Shopify API"""
        creds = credential_manager.get_credentials(self.config.integration_id)
        if not creds or 'access_token' not in creds or 'shop_domain' not in creds:
            return False

        self.shop_domain = creds['shop_domain']

        # Test authentication with a simple API call
        response = await self.make_request("GET", "shop.json")
        return response.success

    async def health_check(self) -> bool:
        """Check Shopify API health"""
        response = await self.make_request("GET", "shop.json")
        return response.success

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get Shopify authentication headers"""
        creds = credential_manager.get_credentials(self.config.integration_id)
        if not creds or 'access_token' not in creds:
            return {}

        return {
            "X-Shopify-Access-Token": creds['access_token'],
            "Content-Type": "application/json"
        }

    async def get_orders(self, criteria: OrderSearchCriteria,
                        consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Get orders from Shopify"""
        params = {
            "limit": min(criteria.limit, 250),  # Shopify limit
            "status": "any"
        }

        if criteria.customer_email:
            # Note: Shopify doesn't support direct email search, would need to get customer first
            pass

        if criteria.date_from:
            params["created_at_min"] = criteria.date_from.isoformat()

        if criteria.date_to:
            params["created_at_max"] = criteria.date_to.isoformat()

        response = await self.make_request("GET", "orders.json", params=params,
                                         consent_context=consent_context)

        if response.success and response.data:
            # Transform Shopify orders to standardized format
            orders = []
            for shopify_order in response.data.get('orders', []):
                standardized_order = self._transform_shopify_order(shopify_order)
                standardized_order.consent_validated = response.consent_validated
                orders.append(standardized_order)

            response.data = {"orders": orders}

        return response

    async def get_order_by_id(self, order_id: str,
                             consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Get specific order by ID"""
        response = await self.make_request("GET", f"orders/{order_id}.json",
                                         consent_context=consent_context)

        if response.success and response.data:
            order = self._transform_shopify_order(response.data.get('order', {}))
            order.consent_validated = response.consent_validated
            response.data = {"order": order}

        return response

    async def update_order_status(self, order_id: str, status: OrderStatus,
                                 consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Update order status (limited in Shopify)"""
        # Shopify doesn't allow direct status updates, but we can cancel orders
        if status == OrderStatus.CANCELLED:
            return await self.make_request("POST", f"orders/{order_id}/cancel.json",
                                         consent_context=consent_context)

        return IntegrationResponse(
            success=False,
            error_message="Shopify doesn't support direct status updates for this status",
            error_code="NOT_SUPPORTED"
        )

    def _transform_shopify_order(self, shopify_order: Dict[str, Any]) -> StandardizedOrder:
        """Transform Shopify order to standardized format"""

        # Map Shopify status to standard status
        status_mapping = {
            "open": OrderStatus.CONFIRMED,
            "closed": OrderStatus.DELIVERED,
            "cancelled": OrderStatus.CANCELLED
        }

        fulfillment_status = shopify_order.get('fulfillment_status')
        if fulfillment_status == 'fulfilled':
            status = OrderStatus.SHIPPED
        elif fulfillment_status == 'partial':
            status = OrderStatus.PROCESSING
        else:
            status = status_mapping.get(shopify_order.get('financial_status', 'pending'), OrderStatus.PENDING)

        return StandardizedOrder(
            order_id=f"shopify_{shopify_order['id']}",
            platform_order_id=str(shopify_order['id']),
            customer_id=str(shopify_order.get('customer', {}).get('id', '')),
            customer_email=shopify_order.get('email', ''),
            status=status,
            total_amount=float(shopify_order.get('total_price', 0)),
            currency=shopify_order.get('currency', 'USD'),
            created_at=datetime.fromisoformat(shopify_order['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(shopify_order['updated_at'].replace('Z', '+00:00')),
            line_items=shopify_order.get('line_items', []),
            billing_address=shopify_order.get('billing_address'),
            shipping_address=shopify_order.get('shipping_address'),
            tracking_numbers=[f.get('tracking_number', '') for f in shopify_order.get('fulfillments', [])
                            if f.get('tracking_number')],
            platform_data=shopify_order,
            data_source="shopify"
        )


class WooCommerceIntegration(BaseIntegration):
    """WooCommerce order management integration"""

    async def authenticate(self) -> bool:
        """Authenticate with WooCommerce API"""
        creds = credential_manager.get_credentials(self.config.integration_id)
        if not creds or 'consumer_key' not in creds or 'consumer_secret' not in creds:
            return False

        # Test with orders endpoint
        response = await self.make_request("GET", "orders", params={"per_page": 1})
        return response.success

    async def health_check(self) -> bool:
        """Check WooCommerce API health"""
        response = await self.make_request("GET", "")
        return response.success

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get WooCommerce authentication headers"""
        creds = credential_manager.get_credentials(self.config.integration_id)
        if not creds:
            return {}

        # WooCommerce uses basic auth with consumer key/secret
        import base64
        auth_string = f"{creds['consumer_key']}:{creds['consumer_secret']}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()

        return {
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/json"
        }

    async def get_orders(self, criteria: OrderSearchCriteria,
                        consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Get orders from WooCommerce"""
        params = {
            "per_page": min(criteria.limit, 100),  # WooCommerce limit
            "page": (criteria.offset // criteria.limit) + 1
        }

        if criteria.customer_email:
            params["customer"] = criteria.customer_email

        if criteria.status:
            params["status"] = criteria.status.value

        if criteria.date_from:
            params["after"] = criteria.date_from.isoformat()

        if criteria.date_to:
            params["before"] = criteria.date_to.isoformat()

        response = await self.make_request("GET", "orders", params=params,
                                         consent_context=consent_context)

        if response.success and response.data:
            orders = []
            for wc_order in response.data:
                standardized_order = self._transform_woocommerce_order(wc_order)
                standardized_order.consent_validated = response.consent_validated
                orders.append(standardized_order)

            response.data = {"orders": orders}

        return response

    def _transform_woocommerce_order(self, wc_order: Dict[str, Any]) -> StandardizedOrder:
        """Transform WooCommerce order to standardized format"""

        status_mapping = {
            "pending": OrderStatus.PENDING,
            "processing": OrderStatus.PROCESSING,
            "on-hold": OrderStatus.CONFIRMED,
            "completed": OrderStatus.DELIVERED,
            "cancelled": OrderStatus.CANCELLED,
            "refunded": OrderStatus.REFUNDED,
            "failed": OrderStatus.CANCELLED
        }

        return StandardizedOrder(
            order_id=f"wc_{wc_order['id']}",
            platform_order_id=str(wc_order['id']),
            customer_id=str(wc_order.get('customer_id', '')),
            customer_email=wc_order.get('billing', {}).get('email', ''),
            status=status_mapping.get(wc_order.get('status'), OrderStatus.PENDING),
            total_amount=float(wc_order.get('total', 0)),
            currency=wc_order.get('currency', 'USD'),
            created_at=datetime.fromisoformat(wc_order['date_created'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(wc_order['date_modified'].replace('Z', '+00:00')),
            line_items=wc_order.get('line_items', []),
            billing_address=wc_order.get('billing'),
            shipping_address=wc_order.get('shipping'),
            platform_data=wc_order,
            data_source="woocommerce"
        )


class BigCommerceIntegration(BaseIntegration):
    """BigCommerce order management integration"""

    async def authenticate(self) -> bool:
        """Authenticate with BigCommerce API"""
        creds = credential_manager.get_credentials(self.config.integration_id)
        if not creds or 'access_token' not in creds or 'store_hash' not in creds:
            return False

        response = await self.make_request("GET", "store")
        return response.success

    async def health_check(self) -> bool:
        """Check BigCommerce API health"""
        response = await self.make_request("GET", "store")
        return response.success

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get BigCommerce authentication headers"""
        creds = credential_manager.get_credentials(self.config.integration_id)
        if not creds:
            return {}

        return {
            "X-Auth-Token": creds['access_token'],
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    async def get_orders(self, criteria: OrderSearchCriteria,
                        consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Get orders from BigCommerce"""
        params = {
            "limit": min(criteria.limit, 250),  # BigCommerce limit
            "page": (criteria.offset // criteria.limit) + 1
        }

        if criteria.customer_email:
            params["email"] = criteria.customer_email

        if criteria.status:
            # Map to BigCommerce status IDs (simplified)
            status_mapping = {
                OrderStatus.PENDING: 1,
                OrderStatus.PROCESSING: 2,
                OrderStatus.SHIPPED: 3,
                OrderStatus.DELIVERED: 4,
                OrderStatus.CANCELLED: 5
            }
            if criteria.status in status_mapping:
                params["status_id"] = status_mapping[criteria.status]

        response = await self.make_request("GET", "orders", params=params,
                                         consent_context=consent_context)

        if response.success and response.data:
            orders = []
            for bc_order in response.data:
                standardized_order = self._transform_bigcommerce_order(bc_order)
                standardized_order.consent_validated = response.consent_validated
                orders.append(standardized_order)

            response.data = {"orders": orders}

        return response

    def _transform_bigcommerce_order(self, bc_order: Dict[str, Any]) -> StandardizedOrder:
        """Transform BigCommerce order to standardized format"""

        # BigCommerce uses status IDs
        status_mapping = {
            1: OrderStatus.PENDING,
            2: OrderStatus.PROCESSING,
            3: OrderStatus.SHIPPED,
            4: OrderStatus.DELIVERED,
            5: OrderStatus.CANCELLED,
            6: OrderStatus.REFUNDED
        }

        return StandardizedOrder(
            order_id=f"bc_{bc_order['id']}",
            platform_order_id=str(bc_order['id']),
            customer_id=str(bc_order.get('customer_id', '')),
            customer_email=bc_order.get('billing_address', {}).get('email', ''),
            status=status_mapping.get(bc_order.get('status_id'), OrderStatus.PENDING),
            total_amount=float(bc_order.get('total_inc_tax', 0)),
            currency=bc_order.get('currency_code', 'USD'),
            created_at=datetime.fromisoformat(bc_order['date_created']),
            updated_at=datetime.fromisoformat(bc_order['date_modified']),
            billing_address=bc_order.get('billing_address'),
            shipping_address=bc_order.get('shipping_addresses', [{}])[0] if bc_order.get('shipping_addresses') else None,
            platform_data=bc_order,
            data_source="bigcommerce"
        )


class OrderManagementOrchestrator:
    """Orchestrator for managing multiple order management integrations"""

    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}

    def register_integration(self, integration_id: str, integration: BaseIntegration):
        """Register an order management integration"""
        self.integrations[integration_id] = integration

    async def search_orders_across_platforms(self, criteria: OrderSearchCriteria,
                                           platforms: Optional[List[str]] = None,
                                           consent_context: Optional[ConsentContext] = None) -> Dict[str, IntegrationResponse]:
        """Search orders across multiple platforms"""

        if platforms is None:
            platforms = list(self.integrations.keys())

        results = {}

        for platform in platforms:
            if platform in self.integrations:
                try:
                    integration = self.integrations[platform]
                    response = await integration.get_orders(criteria, consent_context)
                    results[platform] = response
                except Exception as e:
                    results[platform] = IntegrationResponse(
                        success=False,
                        error_message=str(e),
                        error_code="PLATFORM_ERROR"
                    )

        return results

    async def get_customer_order_history(self, customer_email: str,
                                       platforms: Optional[List[str]] = None,
                                       consent_context: Optional[ConsentContext] = None) -> List[StandardizedOrder]:
        """Get complete order history for a customer across platforms"""

        criteria = OrderSearchCriteria(customer_email=customer_email, limit=100)
        results = await self.search_orders_across_platforms(criteria, platforms, consent_context)

        all_orders = []
        for platform, response in results.items():
            if response.success and response.data:
                orders = response.data.get('orders', [])
                all_orders.extend(orders)

        # Sort by creation date, most recent first
        all_orders.sort(key=lambda x: x.created_at, reverse=True)

        return all_orders

    async def get_order_insights(self, customer_email: str,
                               consent_context: Optional[ConsentContext] = None) -> Dict[str, Any]:
        """Get customer insights from order history"""

        orders = await self.get_customer_order_history(customer_email, consent_context=consent_context)

        if not orders:
            return {"total_orders": 0, "insights": []}

        total_spent = sum(order.total_amount for order in orders)
        avg_order_value = total_spent / len(orders)

        # Recent activity
        recent_orders = [o for o in orders if (datetime.now() - o.created_at).days <= 90]

        # Status distribution
        status_counts = {}
        for order in orders:
            status_counts[order.status.value] = status_counts.get(order.status.value, 0) + 1

        insights = []

        if len(recent_orders) >= 3:
            insights.append("High activity customer - 3+ orders in last 90 days")

        if total_spent > 1000:
            insights.append(f"High value customer - ${total_spent:.2f} total spent")

        if any(order.status == OrderStatus.RETURNED for order in orders):
            insights.append("Has returned items previously")

        return {
            "total_orders": len(orders),
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "recent_orders_90d": len(recent_orders),
            "status_distribution": status_counts,
            "platforms": list(set(order.data_source for order in orders)),
            "insights": insights,
            "last_order_date": orders[0].created_at.isoformat() if orders else None
        }


# Global orchestrator instance
order_orchestrator = OrderManagementOrchestrator()