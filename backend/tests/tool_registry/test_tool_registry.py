import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.tooling import ToolRegistry, SideEffect
from app.tooling.exceptions import ManifestValidationError, OperationNotFoundError, ToolNotFoundError


MANIFEST_V1 = """
tool:
  name: "salesforce.crm"
  version: "1.0.0"
  display_name: "Salesforce CRM"
  description: "CRM operations"
  auth:
    kind: "oauth2"
    scopes: ["api"]
  governance:
    pii: "may_contain"
  rate_limits:
    per_minute: 120
  schemas:
    Contact:
      type: object
      properties:
        id: {type: string}
  operations:
    - op_id: "crm.search_contacts"
      method: "GET"
      path: "/contacts/search"
      side_effect: "read"
      description: "Search contacts"
      args_schema:
        q:
          type: string
          required: true
      returns:
        type: array
        items: {type: object}
      preconditions: []
      postconditions: []
      errors:
        - code: 429
          message: "Rate limit"
    - op_id: "crm.create_opportunity"
      method: "POST"
      path: "/opportunities"
      side_effect: "write"
      description: "Create opportunity"
      args_schema:
        name:
          type: string
          required: true
      returns:
        type: object
      preconditions:
        - "args.name != ''"
      postconditions:
        - "result.id is not None"
      idempotency:
        header: "Idempotency-Key"
      compensation:
        operation: "crm.delete_opportunity"
        args_mapping:
          opportunity_id: "result.id"
      errors: []
"""

MANIFEST_V2 = """
tool:
  name: "salesforce.crm"
  version: "1.1.0"
  display_name: "Salesforce CRM"
  description: "CRM operations v2"
  auth:
    kind: "oauth2"
    scopes: ["api"]
  governance:
    pii: "may_contain"
  rate_limits:
    per_minute: 200
  schemas:
    Contact:
      type: object
      properties:
        id: {type: string}
  operations:
    - op_id: "crm.search_contacts"
      method: "GET"
      path: "/contacts/search"
      side_effect: "read"
      description: "Search contacts with filters"
      args_schema:
        q:
          type: string
          required: true
      returns:
        type: array
        items: {type: object}
    - op_id: "crm.delete_opportunity"
      method: "DELETE"
      path: "/opportunities/{id}"
      side_effect: "destructive"
      description: "Delete opportunity"
      args_schema:
        id:
          type: string
          required: true
      returns:
        type: object
"""

INVALID_MANIFEST = {"tool": {"name": "bad.tool"}}


@pytest_asyncio.fixture
async def registry():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def session_factory():
        return async_session()

    registry = ToolRegistry(session_factory=session_factory)
    try:
        yield registry
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_register_and_get_tool(registry: ToolRegistry):
    manifest = await registry.register_manifest_from_yaml(MANIFEST_V1)

    assert manifest.name == "salesforce.crm"
    assert manifest.version == "1.0.0"
    assert "crm.search_contacts" in manifest.operations

    op = manifest.operations["crm.create_opportunity"]
    assert op.side_effect == SideEffect.WRITE
    assert op.idempotency_header == "Idempotency-Key"
    assert op.preconditions == ["args.name != ''"]
    assert op.compensation and op.compensation["operation"] == "crm.delete_opportunity"


@pytest.mark.asyncio
async def test_get_operation(registry: ToolRegistry):
    await registry.register_manifest_from_yaml(MANIFEST_V1)
    operation = await registry.get_operation("salesforce.crm", "crm.search_contacts")
    assert operation.side_effect == SideEffect.READ
    assert operation.method == "GET"

    with pytest.raises(OperationNotFoundError):
        await registry.get_operation("salesforce.crm", "unknown")


@pytest.mark.asyncio
async def test_register_new_version_updates_latest(registry: ToolRegistry):
    await registry.register_manifest_from_yaml(MANIFEST_V1)
    await registry.register_manifest_from_yaml(MANIFEST_V2)

    latest = await registry.get_tool("salesforce.crm")
    assert latest.version == "1.1.0"
    assert "crm.delete_opportunity" in latest.operations
    assert latest.operations["crm.delete_opportunity"].side_effect == SideEffect.DESTRUCTIVE


@pytest.mark.asyncio
async def test_search_tools(registry: ToolRegistry):
    await registry.register_manifest_from_yaml(MANIFEST_V1)
    results = await registry.search_tools("opportunity")
    assert results
    assert results[0].name == "salesforce.crm"


@pytest.mark.asyncio
async def test_invalid_manifest_raises(registry: ToolRegistry):
    with pytest.raises(ManifestValidationError):
        await registry.register_manifest(INVALID_MANIFEST)


@pytest.mark.asyncio
async def test_get_tool_missing_version(registry: ToolRegistry):
    with pytest.raises(ToolNotFoundError):
        await registry.get_tool("nonexistent")
