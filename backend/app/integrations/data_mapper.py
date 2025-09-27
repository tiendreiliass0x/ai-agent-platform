"""
Data Mapping and Transformation Layer

This module provides data transformation, field mapping, and normalization
capabilities for integrating data from different third-party platforms.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
import asyncio

from ..core.governance import ConsentContext, governance_engine


class DataType(str, Enum):
    """Supported data types for mapping"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    ARRAY = "array"
    OBJECT = "object"
    EMAIL = "email"
    PHONE = "phone"
    CURRENCY = "currency"


class TransformationType(str, Enum):
    """Types of data transformations"""
    DIRECT_MAP = "direct_map"           # Direct field mapping
    CONCAT = "concat"                   # Concatenate multiple fields
    SPLIT = "split"                     # Split field into multiple
    FORMAT = "format"                   # Format using template
    LOOKUP = "lookup"                   # Value lookup/mapping
    COMPUTE = "compute"                 # Custom computation
    CONDITIONAL = "conditional"         # Conditional logic
    NORMALIZE = "normalize"             # Data normalization


@dataclass
class FieldMapping:
    """Mapping configuration for a single field"""
    source_path: str                    # JSONPath to source field
    target_path: str                    # JSONPath to target field
    data_type: DataType
    transformation: TransformationType = TransformationType.DIRECT_MAP

    # Transformation parameters
    format_template: Optional[str] = None        # For FORMAT transformation
    lookup_table: Optional[Dict[str, Any]] = None    # For LOOKUP transformation
    compute_function: Optional[str] = None       # For COMPUTE transformation
    conditional_rules: Optional[List[Dict]] = None   # For CONDITIONAL transformation

    # Validation
    required: bool = False
    default_value: Any = None
    validation_regex: Optional[str] = None

    # Governance
    contains_pii: bool = False
    requires_consent: bool = False


@dataclass
class PlatformSchema:
    """Schema definition for a platform's data structure"""
    platform_id: str
    schema_version: str
    field_mappings: List[FieldMapping] = field(default_factory=list)
    custom_transformations: Dict[str, Callable] = field(default_factory=dict)

    # Metadata
    description: str = ""
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TransformationResult:
    """Result of data transformation operation"""
    success: bool
    transformed_data: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fields_processed: int = 0
    fields_skipped: int = 0
    pii_fields_redacted: List[str] = field(default_factory=list)


class DataMapper:
    """Core data mapping and transformation engine"""

    def __init__(self):
        self.schemas: Dict[str, PlatformSchema] = {}
        self.value_normalizers: Dict[str, Callable] = {}
        self._register_default_normalizers()

    def register_schema(self, schema: PlatformSchema):
        """Register a platform schema"""
        self.schemas[schema.platform_id] = schema

    def _register_default_normalizers(self):
        """Register default data normalizers"""

        def normalize_phone(phone: str) -> str:
            """Normalize phone number format"""
            if not phone:
                return ""
            # Remove all non-digits
            digits = re.sub(r'\D', '', phone)
            # Format as +1-XXX-XXX-XXXX for US numbers
            if len(digits) == 10:
                return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
            return phone

        def normalize_email(email: str) -> str:
            """Normalize email address"""
            if not email:
                return ""
            return email.lower().strip()

        def normalize_currency(amount: Any) -> float:
            """Normalize currency values"""
            if isinstance(amount, (int, float)):
                return float(amount)
            if isinstance(amount, str):
                # Remove currency symbols and commas
                clean_amount = re.sub(r'[^\d.-]', '', amount)
                try:
                    return float(clean_amount)
                except ValueError:
                    return 0.0
            return 0.0

        def normalize_boolean(value: Any) -> bool:
            """Normalize boolean values"""
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            if isinstance(value, (int, float)):
                return bool(value)
            return False

        self.value_normalizers = {
            DataType.PHONE: normalize_phone,
            DataType.EMAIL: normalize_email,
            DataType.CURRENCY: normalize_currency,
            DataType.BOOLEAN: normalize_boolean
        }

    def get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif isinstance(value, list) and key.isdigit():
                try:
                    value = value[int(key)]
                except (IndexError, ValueError):
                    return None
            else:
                return None

        return value

    def set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = path.split('.')
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def apply_transformation(self, mapping: FieldMapping, source_data: Dict[str, Any],
                           consent_context: Optional[ConsentContext] = None) -> Any:
        """Apply transformation to a single field"""

        # Get source value
        source_value = self.get_nested_value(source_data, mapping.source_path)

        # Check for required field
        if mapping.required and source_value is None:
            raise ValueError(f"Required field {mapping.source_path} is missing")

        # Use default if value is None
        if source_value is None:
            source_value = mapping.default_value

        # Check consent for PII fields
        if mapping.contains_pii and consent_context:
            if not governance_engine.validate_consent(consent_context,
                                                    getattr(consent_context, 'allowed_inferences', set())):
                return "[REDACTED]"

        # Apply transformation based on type
        if mapping.transformation == TransformationType.DIRECT_MAP:
            return self._apply_direct_map(source_value, mapping)

        elif mapping.transformation == TransformationType.FORMAT:
            return self._apply_format(source_value, mapping, source_data)

        elif mapping.transformation == TransformationType.LOOKUP:
            return self._apply_lookup(source_value, mapping)

        elif mapping.transformation == TransformationType.CONDITIONAL:
            return self._apply_conditional(source_value, mapping, source_data)

        elif mapping.transformation == TransformationType.COMPUTE:
            return self._apply_compute(source_value, mapping, source_data)

        elif mapping.transformation == TransformationType.NORMALIZE:
            return self._apply_normalize(source_value, mapping)

        else:
            return source_value

    def _apply_direct_map(self, value: Any, mapping: FieldMapping) -> Any:
        """Apply direct mapping with type conversion"""
        if value is None:
            return mapping.default_value

        # Type conversion
        if mapping.data_type == DataType.STRING:
            return str(value)
        elif mapping.data_type == DataType.INTEGER:
            try:
                return int(float(value))  # Handle "123.0" strings
            except (ValueError, TypeError):
                return mapping.default_value or 0
        elif mapping.data_type == DataType.FLOAT:
            try:
                return float(value)
            except (ValueError, TypeError):
                return mapping.default_value or 0.0
        elif mapping.data_type == DataType.BOOLEAN:
            return self.value_normalizers[DataType.BOOLEAN](value)
        elif mapping.data_type == DataType.DATETIME:
            if isinstance(value, str):
                try:
                    # Handle various datetime formats
                    for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            return datetime.strptime(value.replace('Z', ''), fmt)
                        except ValueError:
                            continue
                    # Try ISO format
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    return mapping.default_value
            return value

        return value

    def _apply_format(self, value: Any, mapping: FieldMapping, source_data: Dict[str, Any]) -> str:
        """Apply format template transformation"""
        if not mapping.format_template:
            return str(value)

        # Replace placeholders in template
        template = mapping.format_template

        # Replace {field_name} with values from source data
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template)

        for match in matches:
            field_value = self.get_nested_value(source_data, match)
            if field_value is not None:
                template = template.replace(f'{{{match}}}', str(field_value))

        return template

    def _apply_lookup(self, value: Any, mapping: FieldMapping) -> Any:
        """Apply lookup table transformation"""
        if not mapping.lookup_table:
            return value

        return mapping.lookup_table.get(str(value), mapping.default_value or value)

    def _apply_conditional(self, value: Any, mapping: FieldMapping, source_data: Dict[str, Any]) -> Any:
        """Apply conditional transformation"""
        if not mapping.conditional_rules:
            return value

        for rule in mapping.conditional_rules:
            condition = rule.get('condition')
            result = rule.get('result')

            # Simple condition evaluation
            if self._evaluate_condition(condition, value, source_data):
                return result

        return mapping.default_value or value

    def _apply_compute(self, value: Any, mapping: FieldMapping, source_data: Dict[str, Any]) -> Any:
        """Apply custom computation"""
        if not mapping.compute_function:
            return value

        # Basic arithmetic operations
        if mapping.compute_function.startswith('sum:'):
            fields = mapping.compute_function.split(':')[1].split(',')
            total = 0
            for field in fields:
                field_value = self.get_nested_value(source_data, field.strip())
                if isinstance(field_value, (int, float)):
                    total += field_value
            return total

        elif mapping.compute_function.startswith('concat:'):
            fields = mapping.compute_function.split(':')[1].split(',')
            parts = []
            for field in fields:
                field_value = self.get_nested_value(source_data, field.strip())
                if field_value is not None:
                    parts.append(str(field_value))
            return ' '.join(parts)

        return value

    def _apply_normalize(self, value: Any, mapping: FieldMapping) -> Any:
        """Apply normalization"""
        if mapping.data_type in self.value_normalizers:
            return self.value_normalizers[mapping.data_type](value)
        return value

    def _evaluate_condition(self, condition: str, value: Any, source_data: Dict[str, Any]) -> bool:
        """Evaluate simple conditional expressions"""
        if not condition:
            return False

        # Simple condition formats: "field == value", "value > 10", etc.
        operators = ['==', '!=', '>', '<', '>=', '<=', 'in', 'not_in']

        for op in operators:
            if op in condition:
                left, right = condition.split(op, 1)
                left = left.strip()
                right = right.strip().strip('"\'')

                # Get left value (could be field reference or literal)
                if left.startswith('{') and left.endswith('}'):
                    left_value = self.get_nested_value(source_data, left[1:-1])
                else:
                    left_value = value

                # Evaluate condition
                if op == '==':
                    return str(left_value) == right
                elif op == '!=':
                    return str(left_value) != right
                elif op == '>':
                    try:
                        return float(left_value) > float(right)
                    except (ValueError, TypeError):
                        return False
                elif op == '<':
                    try:
                        return float(left_value) < float(right)
                    except (ValueError, TypeError):
                        return False
                # Add more operators as needed

        return False

    async def transform_data(self, platform_id: str, source_data: Dict[str, Any],
                           target_schema: Optional[str] = None,
                           consent_context: Optional[ConsentContext] = None) -> TransformationResult:
        """Transform data from one platform to standardized format"""

        if platform_id not in self.schemas:
            return TransformationResult(
                success=False,
                errors=[f"Schema not found for platform: {platform_id}"]
            )

        schema = self.schemas[platform_id]
        result = TransformationResult(success=True)
        transformed_data = {}

        for mapping in schema.field_mappings:
            try:
                # Check consent requirements
                if mapping.requires_consent and consent_context:
                    from ..core.governance import ConsentScope
                    if not governance_engine.validate_consent(consent_context, ConsentScope.USE_CONVERSATION_HISTORY):
                        result.fields_skipped += 1
                        continue

                # Apply transformation
                transformed_value = self.apply_transformation(mapping, source_data, consent_context)

                # Validate if regex provided
                if mapping.validation_regex and transformed_value is not None:
                    if not re.match(mapping.validation_regex, str(transformed_value)):
                        result.warnings.append(f"Field {mapping.target_path} failed validation")

                # Set in target
                self.set_nested_value(transformed_data, mapping.target_path, transformed_value)

                result.fields_processed += 1

                # Track PII redaction
                if mapping.contains_pii and transformed_value == "[REDACTED]":
                    result.pii_fields_redacted.append(mapping.target_path)

            except Exception as e:
                result.errors.append(f"Error transforming {mapping.source_path}: {str(e)}")
                result.success = False

        result.transformed_data = transformed_data
        return result

    def create_reverse_mapping(self, platform_id: str) -> Optional[PlatformSchema]:
        """Create reverse mapping schema for data sync back to platform"""
        if platform_id not in self.schemas:
            return None

        original_schema = self.schemas[platform_id]
        reverse_mappings = []

        for mapping in original_schema.field_mappings:
            # Swap source and target paths
            reverse_mapping = FieldMapping(
                source_path=mapping.target_path,
                target_path=mapping.source_path,
                data_type=mapping.data_type,
                transformation=mapping.transformation,
                format_template=mapping.format_template,
                lookup_table=mapping.lookup_table,
                compute_function=mapping.compute_function,
                conditional_rules=mapping.conditional_rules,
                required=mapping.required,
                default_value=mapping.default_value,
                validation_regex=mapping.validation_regex,
                contains_pii=mapping.contains_pii,
                requires_consent=mapping.requires_consent
            )
            reverse_mappings.append(reverse_mapping)

        return PlatformSchema(
            platform_id=f"{platform_id}_reverse",
            schema_version=original_schema.schema_version,
            field_mappings=reverse_mappings,
            description=f"Reverse mapping for {platform_id}"
        )


class SchemaRegistry:
    """Registry for managing platform schemas"""

    def __init__(self):
        self.mapper = DataMapper()
        self._load_default_schemas()

    def _load_default_schemas(self):
        """Load default schemas for popular platforms"""

        # Shopify order schema
        shopify_order_mappings = [
            FieldMapping("id", "order_id", DataType.STRING, required=True),
            FieldMapping("email", "customer_email", DataType.EMAIL, contains_pii=True),
            FieldMapping("total_price", "total_amount", DataType.CURRENCY),
            FieldMapping("currency", "currency", DataType.STRING),
            FieldMapping("created_at", "created_at", DataType.DATETIME),
            FieldMapping("financial_status", "payment_status", DataType.STRING,
                        transformation=TransformationType.LOOKUP,
                        lookup_table={
                            "paid": "completed",
                            "pending": "pending",
                            "refunded": "refunded",
                            "partially_refunded": "partially_refunded"
                        }),
            FieldMapping("fulfillment_status", "fulfillment_status", DataType.STRING,
                        transformation=TransformationType.LOOKUP,
                        lookup_table={
                            "fulfilled": "shipped",
                            "partial": "processing",
                            "restocked": "cancelled"
                        })
        ]

        shopify_schema = PlatformSchema(
            platform_id="shopify_orders",
            schema_version="1.0",
            field_mappings=shopify_order_mappings,
            description="Shopify order data mapping"
        )

        # WooCommerce product schema
        woocommerce_product_mappings = [
            FieldMapping("id", "product_id", DataType.STRING, required=True),
            FieldMapping("name", "title", DataType.STRING, required=True),
            FieldMapping("description", "description", DataType.STRING),
            FieldMapping("price", "price", DataType.CURRENCY),
            FieldMapping("regular_price", "compare_at_price", DataType.CURRENCY),
            FieldMapping("sku", "sku", DataType.STRING),
            FieldMapping("stock_quantity", "inventory_quantity", DataType.INTEGER),
            FieldMapping("status", "status", DataType.STRING,
                        transformation=TransformationType.LOOKUP,
                        lookup_table={
                            "publish": "active",
                            "draft": "draft",
                            "private": "inactive"
                        })
        ]

        woocommerce_schema = PlatformSchema(
            platform_id="woocommerce_products",
            schema_version="1.0",
            field_mappings=woocommerce_product_mappings,
            description="WooCommerce product data mapping"
        )

        # Register schemas
        self.mapper.register_schema(shopify_schema)
        self.mapper.register_schema(woocommerce_schema)

    def get_mapper(self) -> DataMapper:
        """Get the data mapper instance"""
        return self.mapper

    def register_custom_schema(self, schema: PlatformSchema):
        """Register a custom platform schema"""
        self.mapper.register_schema(schema)

    def list_schemas(self) -> List[str]:
        """List all registered schema IDs"""
        return list(self.mapper.schemas.keys())


# Global schema registry instance
schema_registry = SchemaRegistry()