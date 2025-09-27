"""
Product Catalog Integration Service

This module provides connectors for product catalog systems including
e-commerce platforms, PIM systems, and custom product APIs.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from .base import (
    BaseIntegration, IntegrationConfig, IntegrationCredentials, IntegrationResponse,
    IntegrationType, AuthMethod, credential_manager
)
from ..core.governance import ConsentContext


class ProductStatus(str, Enum):
    """Product availability status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ARCHIVED = "archived"
    OUT_OF_STOCK = "out_of_stock"


class ProductType(str, Enum):
    """Product type classification"""
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    BUNDLE = "bundle"
    VARIANT = "variant"


@dataclass
class ProductVariant:
    """Product variant information"""
    variant_id: str
    sku: str
    title: str
    price: float
    compare_at_price: Optional[float] = None
    inventory_quantity: int = 0
    attributes: Dict[str, str] = field(default_factory=dict)  # color, size, etc.
    image_url: Optional[str] = None


@dataclass
class ProductInventory:
    """Product inventory information"""
    sku: str
    quantity_available: int
    quantity_reserved: int = 0
    quantity_on_order: int = 0
    reorder_point: int = 0
    warehouse_locations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class StandardizedProduct:
    """Standardized product representation across platforms"""
    product_id: str
    platform_product_id: str
    title: str
    description: str
    price: float

    # Categorization
    product_type: ProductType = ProductType.PHYSICAL
    status: ProductStatus = ProductStatus.ACTIVE
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Pricing
    compare_at_price: Optional[float] = None
    currency: str = "USD"

    # Inventory
    sku: Optional[str] = None
    inventory: Optional[ProductInventory] = None

    # Variants
    variants: List[ProductVariant] = field(default_factory=list)

    # Media
    images: List[str] = field(default_factory=list)

    # SEO and metadata
    handle: Optional[str] = None  # URL slug
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Platform-specific data
    platform_data: Dict[str, Any] = field(default_factory=dict)

    # Governance
    data_source: str = ""
    consent_validated: bool = False


@dataclass
class ProductSearchCriteria:
    """Search criteria for product queries"""
    query: Optional[str] = None
    category: Optional[str] = None
    status: Optional[ProductStatus] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    in_stock_only: bool = False
    tags: Optional[List[str]] = None
    limit: int = 50
    offset: int = 0


class ShopifyProductCatalog(BaseIntegration):
    """Shopify product catalog integration"""

    async def authenticate(self) -> bool:
        """Authenticate with Shopify API"""
        creds = credential_manager.get_credentials(self.config.integration_id)
        if not creds or 'access_token' not in creds or 'shop_domain' not in creds:
            return False

        response = await self.make_request("GET", "shop.json")
        return response.success

    async def health_check(self) -> bool:
        """Check Shopify API health"""
        response = await self.make_request("GET", "products.json", params={"limit": 1})
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

    async def search_products(self, criteria: ProductSearchCriteria,
                            consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Search products in Shopify"""
        params = {
            "limit": min(criteria.limit, 250),  # Shopify limit
            "published_status": "published" if criteria.status == ProductStatus.ACTIVE else "any"
        }

        if criteria.query:
            params["title"] = criteria.query

        response = await self.make_request("GET", "products.json", params=params,
                                         consent_context=consent_context)

        if response.success and response.data:
            products = []
            for shopify_product in response.data.get('products', []):
                standardized_product = self._transform_shopify_product(shopify_product)
                standardized_product.consent_validated = response.consent_validated
                products.append(standardized_product)

            response.data = {"products": products}

        return response

    async def get_product_by_id(self, product_id: str,
                              consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Get specific product by ID"""
        response = await self.make_request("GET", f"products/{product_id}.json",
                                         consent_context=consent_context)

        if response.success and response.data:
            product = self._transform_shopify_product(response.data.get('product', {}))
            product.consent_validated = response.consent_validated
            response.data = {"product": product}

        return response

    async def get_product_recommendations(self, product_id: str,
                                        consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Get product recommendations (Shopify doesn't have native recommendations)"""
        # Get product to find related by tags/category
        product_response = await self.get_product_by_id(product_id, consent_context)

        if not product_response.success:
            return product_response

        product = product_response.data.get('product')
        if not product or not product.tags:
            return IntegrationResponse(
                success=True,
                data={"recommendations": []}
            )

        # Search for products with similar tags
        criteria = ProductSearchCriteria(tags=product.tags[:3], limit=10)
        return await self.search_products(criteria, consent_context)

    def _transform_shopify_product(self, shopify_product: Dict[str, Any]) -> StandardizedProduct:
        """Transform Shopify product to standardized format"""

        # Status mapping
        status = ProductStatus.ACTIVE if shopify_product.get('status') == 'active' else ProductStatus.INACTIVE

        # Extract variants
        variants = []
        for variant in shopify_product.get('variants', []):
            variants.append(ProductVariant(
                variant_id=str(variant['id']),
                sku=variant.get('sku', ''),
                title=variant.get('title', ''),
                price=float(variant.get('price', 0)),
                compare_at_price=float(variant['compare_at_price']) if variant.get('compare_at_price') else None,
                inventory_quantity=variant.get('inventory_quantity', 0),
                attributes={
                    'option1': variant.get('option1'),
                    'option2': variant.get('option2'),
                    'option3': variant.get('option3')
                }
            ))

        # Extract images
        images = [img['src'] for img in shopify_product.get('images', [])]

        return StandardizedProduct(
            product_id=f"shopify_{shopify_product['id']}",
            platform_product_id=str(shopify_product['id']),
            title=shopify_product.get('title', ''),
            description=shopify_product.get('body_html', ''),
            product_type=ProductType.PHYSICAL,  # Default, could be enhanced
            status=status,
            categories=[shopify_product.get('product_type', '')] if shopify_product.get('product_type') else [],
            tags=shopify_product.get('tags', '').split(',') if shopify_product.get('tags') else [],
            price=float(variants[0].price) if variants else 0.0,
            sku=variants[0].sku if variants else None,
            variants=variants,
            images=images,
            handle=shopify_product.get('handle'),
            seo_title=shopify_product.get('title'),
            seo_description=shopify_product.get('meta_description'),
            created_at=datetime.fromisoformat(shopify_product['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(shopify_product['updated_at'].replace('Z', '+00:00')),
            platform_data=shopify_product,
            data_source="shopify"
        )


class WooCommerceProductCatalog(BaseIntegration):
    """WooCommerce product catalog integration"""

    async def authenticate(self) -> bool:
        """Authenticate with WooCommerce API"""
        creds = credential_manager.get_credentials(self.config.integration_id)
        if not creds or 'consumer_key' not in creds or 'consumer_secret' not in creds:
            return False

        response = await self.make_request("GET", "products", params={"per_page": 1})
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

        import base64
        auth_string = f"{creds['consumer_key']}:{creds['consumer_secret']}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()

        return {
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/json"
        }

    async def search_products(self, criteria: ProductSearchCriteria,
                            consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Search products in WooCommerce"""
        params = {
            "per_page": min(criteria.limit, 100),
            "page": (criteria.offset // criteria.limit) + 1
        }

        if criteria.query:
            params["search"] = criteria.query

        if criteria.status:
            status_mapping = {
                ProductStatus.ACTIVE: "publish",
                ProductStatus.DRAFT: "draft",
                ProductStatus.INACTIVE: "private"
            }
            params["status"] = status_mapping.get(criteria.status, "publish")

        if criteria.category:
            params["category"] = criteria.category

        if criteria.in_stock_only:
            params["stock_status"] = "instock"

        response = await self.make_request("GET", "products", params=params,
                                         consent_context=consent_context)

        if response.success and response.data:
            products = []
            for wc_product in response.data:
                standardized_product = self._transform_woocommerce_product(wc_product)
                standardized_product.consent_validated = response.consent_validated
                products.append(standardized_product)

            response.data = {"products": products}

        return response

    def _transform_woocommerce_product(self, wc_product: Dict[str, Any]) -> StandardizedProduct:
        """Transform WooCommerce product to standardized format"""

        status_mapping = {
            "publish": ProductStatus.ACTIVE,
            "draft": ProductStatus.DRAFT,
            "private": ProductStatus.INACTIVE
        }

        # Extract variants for variable products
        variants = []
        if wc_product.get('type') == 'variable' and wc_product.get('variations'):
            # Would need additional API calls to get variation details
            pass

        # Extract categories
        categories = [cat['name'] for cat in wc_product.get('categories', [])]

        # Extract images
        images = [img['src'] for img in wc_product.get('images', [])]

        return StandardizedProduct(
            product_id=f"wc_{wc_product['id']}",
            platform_product_id=str(wc_product['id']),
            title=wc_product.get('name', ''),
            description=wc_product.get('description', ''),
            product_type=ProductType.PHYSICAL,  # Could be enhanced based on WC type
            status=status_mapping.get(wc_product.get('status'), ProductStatus.ACTIVE),
            categories=categories,
            tags=[tag['name'] for tag in wc_product.get('tags', [])],
            price=float(wc_product.get('price', 0)),
            compare_at_price=float(wc_product['regular_price']) if wc_product.get('regular_price') else None,
            sku=wc_product.get('sku'),
            variants=variants,
            images=images,
            handle=wc_product.get('slug'),
            seo_title=wc_product.get('name'),
            seo_description=wc_product.get('short_description'),
            created_at=datetime.fromisoformat(wc_product['date_created'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(wc_product['date_modified'].replace('Z', '+00:00')),
            platform_data=wc_product,
            data_source="woocommerce"
        )


class ProductRecommendationEngine:
    """AI-powered product recommendation engine"""

    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}

    def register_integration(self, integration_id: str, integration: BaseIntegration):
        """Register a product catalog integration"""
        self.integrations[integration_id] = integration

    async def get_personalized_recommendations(self, customer_email: str,
                                             context: Dict[str, Any] = None,
                                             consent_context: Optional[ConsentContext] = None) -> List[StandardizedProduct]:
        """Get personalized product recommendations for a customer"""

        # This would integrate with the order management system to get purchase history
        from .order_management import order_orchestrator

        order_insights = await order_orchestrator.get_order_insights(customer_email, consent_context)

        recommendations = []

        # Simple recommendation logic based on purchase history
        if order_insights.get('total_orders', 0) > 0:
            # Get products from categories the customer has purchased from
            for platform_id, integration in self.integrations.items():
                try:
                    # Search for popular products
                    criteria = ProductSearchCriteria(limit=10)
                    response = await integration.search_products(criteria, consent_context)

                    if response.success and response.data:
                        products = response.data.get('products', [])
                        recommendations.extend(products[:3])  # Take top 3 from each platform

                except Exception:
                    continue

        return recommendations[:10]  # Return top 10 recommendations

    async def get_product_alternatives(self, product_id: str, platform_id: str,
                                     consent_context: Optional[ConsentContext] = None) -> List[StandardizedProduct]:
        """Get alternative products for a given product"""

        if platform_id not in self.integrations:
            return []

        integration = self.integrations[platform_id]

        # Get the original product
        product_response = await integration.get_product_by_id(product_id.replace(f"{platform_id}_", ""), consent_context)

        if not product_response.success:
            return []

        product = product_response.data.get('product')
        if not product:
            return []

        # Search for similar products based on categories and tags
        criteria = ProductSearchCriteria(
            category=product.categories[0] if product.categories else None,
            tags=product.tags[:3] if product.tags else None,
            price_min=product.price * 0.7,  # Similar price range
            price_max=product.price * 1.3,
            limit=20
        )

        response = await integration.search_products(criteria, consent_context)

        if response.success and response.data:
            alternatives = response.data.get('products', [])
            # Remove the original product and return alternatives
            return [p for p in alternatives if p.product_id != product.product_id][:10]

        return []

    async def search_products_intelligent(self, query: str, customer_context: Dict[str, Any] = None,
                                        consent_context: Optional[ConsentContext] = None) -> Dict[str, List[StandardizedProduct]]:
        """Intelligent product search across all platforms with customer context"""

        results = {}

        # Enhanced search criteria based on customer context
        criteria = ProductSearchCriteria(query=query, limit=20)

        # Adjust search based on customer preferences/history
        if customer_context:
            if customer_context.get('price_preference') == 'budget':
                criteria.price_max = 50.0
            elif customer_context.get('price_preference') == 'premium':
                criteria.price_min = 100.0

        for platform_id, integration in self.integrations.items():
            try:
                response = await integration.search_products(criteria, consent_context)
                if response.success and response.data:
                    results[platform_id] = response.data.get('products', [])
                else:
                    results[platform_id] = []
            except Exception:
                results[platform_id] = []

        return results


# Global recommendation engine instance
product_recommendation_engine = ProductRecommendationEngine()