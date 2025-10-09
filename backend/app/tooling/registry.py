from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

import yaml
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db_session
from ..models.tool_registry import Tool, ToolManifest as ToolManifestModel, ToolOperation
from .exceptions import (
    ManifestValidationError,
    OperationNotFoundError,
    ToolNotFoundError,
    ToolRegistryError,
)
from .models import OperationSpec, SideEffect, ToolManifest
from .utils import hash_dict


class ToolRegistry:
    """Persistent registry for tool manifests and operations."""

    def __init__(
        self,
        session_factory: Callable[[], Any] = get_db_session,
    ) -> None:
        self._session_factory = session_factory
        self._resolved_factory: Optional[Callable[[], Any]] = None

    async def register_manifest_from_yaml(
        self,
        manifest_yaml: str,
        source: str | None = None,
    ) -> ToolManifest:
        data = yaml.safe_load(manifest_yaml)
        return await self.register_manifest(data, source=source or "<yaml>")

    async def register_manifest(
        self,
        manifest: Dict[str, Any],
        source: str | None = None,
    ) -> ToolManifest:
        tool_data = manifest.get("tool")
        if not tool_data:
            raise ManifestValidationError("Manifest missing 'tool' section")

        required_fields = ["name", "version", "display_name", "description", "operations"]
        for field in required_fields:
            if field not in tool_data:
                raise ManifestValidationError(f"Manifest missing required field: tool.{field}")

        operations_data = tool_data.get("operations") or []
        if not operations_data:
            raise ManifestValidationError("Manifest must declare at least one operation")

        self._validate_operations(operations_data)

        session = await self._acquire_session()
        try:
            tool = await self._upsert_tool(session, tool_data)
            manifest_model = await self._get_or_create_manifest(
                session, tool, tool_data["version"], manifest
            )

            for op_data in operations_data:
                args_schema = op_data.get("args_schema", {})
                args_hash = hash_dict(args_schema) if args_schema else None
                await self._upsert_operation(session, manifest_model.id, op_data, args_hash)

            await self._deactivate_other_manifests(session, tool.id, manifest_model.id)

            tool.latest_version = tool_data["version"]
            await session.commit()

        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

        await self._clear_tool_cache(tool_data["name"])
        return await self.get_tool(tool_data["name"], version=tool_data["version"])

    async def get_tool(self, name: str, version: Optional[str] = None) -> ToolManifest:
        session = await self._acquire_session()
        try:
            result = await session.execute(select(Tool).where(Tool.name == name))
            tool_model = result.scalar_one_or_none()
            if not tool_model:
                raise ToolNotFoundError(f"Tool '{name}' not registered")

            manifest_model = await self._select_manifest(session, tool_model, version)
            if not manifest_model:
                raise ToolNotFoundError(
                    f"No manifest found for tool '{name}' and version '{version or tool_model.latest_version}'"
                )

            operations_models = (
                await session.execute(
                    select(ToolOperation).where(ToolOperation.manifest_id == manifest_model.id)
                )
            ).scalars().all()

            return self._build_manifest(tool_model, manifest_model, operations_models)
        finally:
            await session.close()

    async def get_operation(
        self,
        tool_name: str,
        op_id: str,
        version: Optional[str] = None,
    ) -> OperationSpec:
        manifest = await self.get_tool(tool_name, version=version)
        try:
            return manifest.operations[op_id]
        except KeyError as exc:
            raise OperationNotFoundError(
                f"Operation '{op_id}' not found for tool '{tool_name}'"
            ) from exc

    async def search_tools(self, query: str, limit: int = 5) -> List[ToolManifest]:
        session = await self._acquire_session()
        try:
            stmt = select(Tool)
            tool_models = (await session.execute(stmt)).scalars().all()

            query_lower = query.lower()
            results: List[ToolManifest] = []
            for tool_model in tool_models:
                manifest_model = await self._select_manifest(session, tool_model, None)
                if not manifest_model:
                    continue

                operations_models = (
                    await session.execute(
                        select(ToolOperation).where(ToolOperation.manifest_id == manifest_model.id)
                    )
                ).scalars().all()

                manifest = self._build_manifest(tool_model, manifest_model, operations_models)
                haystack = self._build_search_corpus(manifest)
                if query_lower in haystack:
                    results.append(manifest)
                    if len(results) >= limit:
                        break

            return results
        finally:
            await session.close()

    async def list_tools(self) -> List[ToolManifest]:
        session = await self._acquire_session()
        try:
            tool_models = (await session.execute(select(Tool))).scalars().all()
            manifests: List[ToolManifest] = []
            for tool_model in tool_models:
                manifest_model = await self._select_manifest(session, tool_model, None)
                if not manifest_model:
                    continue
                operations_models = (
                    await session.execute(
                        select(ToolOperation).where(ToolOperation.manifest_id == manifest_model.id)
                    )
                ).scalars().all()
                manifests.append(self._build_manifest(tool_model, manifest_model, operations_models))
            return manifests
        finally:
            await session.close()

    def _build_search_corpus(self, manifest: ToolManifest) -> str:
        operation_text = "\n".join(
            f"{op.op_id} {op.description}" for op in manifest.operations.values()
        )
        corpus = f"{manifest.name} {manifest.display_name} {manifest.description}\n{operation_text}"
        return corpus.lower()

    def _validate_operations(self, operations: List[Dict[str, Any]]) -> None:
        seen = set()
        valid_side_effects = {item.value for item in SideEffect}

        for operation in operations:
            for required in ("op_id", "method", "path", "side_effect"):
                if required not in operation:
                    raise ManifestValidationError(f"Operation missing required field: {required}")

            op_id = operation["op_id"]
            if op_id in seen:
                raise ManifestValidationError(f"Duplicate operation id: {op_id}")
            seen.add(op_id)

            side_effect = operation["side_effect"].lower()
            if side_effect not in valid_side_effects:
                raise ManifestValidationError(
                    f"Invalid side_effect '{side_effect}' for operation '{op_id}'"
                )
            operation["side_effect"] = side_effect

            try:
                json.dumps(operation.get("args_schema", {}))
                json.dumps(operation.get("returns", {}))
            except TypeError as exc:
                raise ManifestValidationError(
                    f"Schemas for operation '{op_id}' must be JSON serializable"
                ) from exc

    def _build_manifest(
        self,
        tool_model: Tool,
        manifest_model: ToolManifestModel,
        operations_models: List[ToolOperation],
    ) -> ToolManifest:
        operations: Dict[str, OperationSpec] = {}
        for op_model in operations_models:
            side_effect = SideEffect(op_model.side_effect)
            operations[op_model.op_id] = OperationSpec(
                op_id=op_model.op_id,
                method=op_model.method,
                path=op_model.path,
                side_effect=side_effect,
                description=op_model.description or "",
                args_schema=op_model.args_schema or {},
                returns=op_model.returns_schema or {},
                preconditions=op_model.preconditions or [],
                postconditions=op_model.postconditions or [],
                idempotency_header=op_model.idempotency_header,
                requires_approval=op_model.requires_approval,
                compensation=op_model.compensation or None,
                errors=op_model.errors or [],
            )

        raw_manifest = manifest_model.manifest or {}
        return ToolManifest(
            name=tool_model.name,
            version=manifest_model.version,
            display_name=tool_model.display_name,
            description=tool_model.description or "",
            auth=tool_model.auth or {},
            operations=operations,
            schemas=tool_model.schemas or {},
            rate_limits=tool_model.rate_limits or {},
            governance=tool_model.governance or {},
            raw=raw_manifest,
        )

    async def _upsert_tool(self, session: AsyncSession, tool_data: Dict[str, Any]) -> Tool:
        result = await session.execute(select(Tool).where(Tool.name == tool_data["name"]))
        tool = result.scalar_one_or_none()
        if tool is None:
            tool = Tool(
                name=tool_data["name"],
                display_name=tool_data["display_name"],
                description=tool_data.get("description", ""),
                auth=tool_data.get("auth", {}),
                governance=tool_data.get("governance", {}),
                rate_limits=tool_data.get("rate_limits", {}),
                schemas=tool_data.get("schemas", {}),
            )
            session.add(tool)
            await session.flush()
        else:
            tool.display_name = tool_data["display_name"]
            tool.description = tool_data.get("description", "")
            tool.auth = tool_data.get("auth", {})
            tool.governance = tool_data.get("governance", {})
            tool.rate_limits = tool_data.get("rate_limits", {})
            tool.schemas = tool_data.get("schemas", {})
        return tool

    async def _get_or_create_manifest(
        self,
        session: AsyncSession,
        tool: Tool,
        version: str,
        manifest_payload: Dict[str, Any],
    ) -> ToolManifestModel:
        result = await session.execute(
            select(ToolManifestModel).where(
                ToolManifestModel.tool_id == tool.id,
                ToolManifestModel.version == version,
            )
        )
        manifest_model = result.scalar_one_or_none()
        if manifest_model is None:
            manifest_model = ToolManifestModel(
                tool_id=tool.id,
                version=version,
                is_active=True,
                manifest=manifest_payload,
            )
            session.add(manifest_model)
            await session.flush()
        else:
            manifest_model.manifest = manifest_payload
            manifest_model.is_active = True
        return manifest_model

    async def _upsert_operation(
        self,
        session: AsyncSession,
        manifest_id: int,
        op_data: Dict[str, Any],
        args_hash: Optional[str],
    ) -> None:
        result = await session.execute(
            select(ToolOperation).where(
                ToolOperation.manifest_id == manifest_id,
                ToolOperation.op_id == op_data["op_id"],
            )
        )
        operation_model = result.scalar_one_or_none()
        if operation_model is None:
            operation_model = ToolOperation(
                manifest_id=manifest_id,
                op_id=op_data["op_id"],
            )
            session.add(operation_model)

        operation_model.method = op_data["method"].upper()
        operation_model.path = op_data["path"]
        operation_model.side_effect = op_data["side_effect"].lower()
        operation_model.description = op_data.get("description", "")
        operation_model.args_schema = op_data.get("args_schema", {})
        operation_model.returns_schema = op_data.get("returns", {})
        operation_model.preconditions = op_data.get("preconditions", [])
        operation_model.postconditions = op_data.get("postconditions", [])
        operation_model.idempotency_header = (
            op_data.get("idempotency", {}).get("header") if op_data.get("idempotency") else None
        )
        operation_model.requires_approval = op_data.get("requires_approval", False)
        operation_model.compensation = op_data.get("compensation", {})
        operation_model.errors = op_data.get("errors", [])
        operation_model.args_schema_hash = args_hash

    async def _deactivate_other_manifests(
        self,
        session: AsyncSession,
        tool_id: int,
        active_manifest_id: int,
    ) -> None:
        await session.execute(
            update(ToolManifestModel)
            .where(
                ToolManifestModel.tool_id == tool_id,
                ToolManifestModel.id != active_manifest_id,
            )
            .values(is_active=False)
        )

    async def _clear_tool_cache(self, tool_name: str) -> None:
        # Placeholder for future caching layer (Redis, etc.)
        return None

    async def _acquire_session(self) -> AsyncSession:
        factory = await self._resolve_factory()
        session = factory()
        if asyncio.iscoroutine(session):
            session = await session
        if not isinstance(session, AsyncSession):
            raise ToolRegistryError("Session factory did not return an AsyncSession")
        return session

    async def _resolve_factory(self) -> Callable[[], Any]:
        if self._resolved_factory is not None:
            return self._resolved_factory
        factory = self._session_factory
        if hasattr(factory, "__anext__"):
            factory = await factory.__anext__()
        self._resolved_factory = factory
        return factory
