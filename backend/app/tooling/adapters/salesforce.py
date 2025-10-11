from __future__ import annotations

from typing import Dict

from .base import ToolAdapter, ToolExecutionContext
from app.tooling.models import OperationSpec, ToolManifest


class SalesforceCRMAdapter:
    """Adapter translating generic CRM operations to Salesforce API calls."""

    def __init__(self, config: Dict[str, str]) -> None:
        self.instance_url = config.get("instance_url", "https://example.salesforce.com")

    async def execute(
        self,
        manifest: ToolManifest,
        operation: OperationSpec,
        args: Dict[str, object],
        context: ToolExecutionContext,
    ) -> Dict[str, object]:
        # Placeholder implementation; would call Salesforce REST API here.
        return {
            "adapter": "salesforce",
            "operation": operation.op_id,
            "args": args,
        }


def register_salesforce_adapter(registry, config_resolver):
    def factory(metadata):
        provider = metadata.get("integrations", {}).get("crm", {})
        if provider.get("adapter") != "salesforce":
            return None
        return SalesforceCRMAdapter(provider)

    registry.register("crm", factory)
