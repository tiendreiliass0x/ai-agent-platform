from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, Set

import yaml

from app.tooling.exceptions import ToolNotFoundError
from app.tooling.registry import ToolRegistry

_DEFAULT_MANIFESTS = [
    "crm_tool.yaml",
    "email_tool.yaml",
]

_REGISTERED_TOOLS: Set[str] = set()
_LOAD_LOCK = asyncio.Lock()


async def ensure_default_tools_registered(
    registry: ToolRegistry,
    manifests_dir: Path | None = None,
) -> None:
    """Ensure default tool manifests are registered in the registry."""
    async with _LOAD_LOCK:
        manifests_path = manifests_dir or Path(__file__).parent / "manifests"
        if not manifests_path.exists():
            return

        for manifest_file in _iter_manifest_files(manifests_path):
            try:
                tool_name = _read_manifest_tool_name(manifest_file)
            except Exception:
                continue

            if tool_name in _REGISTERED_TOOLS:
                continue

            try:
                await registry.get_tool(tool_name)
            except ToolNotFoundError:
                with open(manifest_file, "r", encoding="utf-8") as handle:
                    manifest_yaml = handle.read()
                await registry.register_manifest_from_yaml(
                    manifest_yaml=manifest_yaml,
                    source=str(manifest_file),
                )
            _REGISTERED_TOOLS.add(tool_name)


def _iter_manifest_files(base_path: Path) -> Iterable[Path]:
    if not base_path.exists():
        return []
    selected: list[Path] = []
    for filename in _DEFAULT_MANIFESTS:
        candidate = base_path / filename
        if candidate.exists():
            selected.append(candidate)
    return selected


def _read_manifest_tool_name(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    tool = data.get("tool") if isinstance(data, dict) else {}
    name = tool.get("name")
    if not name:
        raise ValueError(f"Manifest {path} missing tool.name")
    return str(name)
