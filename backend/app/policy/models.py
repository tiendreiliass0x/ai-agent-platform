from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Set

from app.tooling.models import ToolManifest


@dataclass
class MaskedTool:
    """Wrapper describing tool visibility for policy masking."""

    manifest: ToolManifest
    visible: bool
    visible_operations: Set[str]

    def copy(self) -> "MaskedTool":
        # Shallow copy manifest (operations reused)
        return MaskedTool(
            manifest=replace(self.manifest),
            visible=self.visible,
            visible_operations=set(self.visible_operations),
        )
