"""
Tool Registration Script

Registers tool manifests from YAML files into the database-backed registry.
Run this script to load CRM, Email, and other tool definitions.
"""

import asyncio
import pathlib
from app.services.tool_registry import tool_registry


async def register_all_tools():
    """Register all tool manifests from the manifests directory"""

    manifests_dir = pathlib.Path(__file__).parent / "manifests"

    if not manifests_dir.exists():
        print(f"Manifests directory not found: {manifests_dir}")
        return

    registered_count = 0
    failed_count = 0

    for manifest_file in manifests_dir.glob("*.yaml"):
        print(f"\nRegistering tool from: {manifest_file.name}")

        try:
            with open(manifest_file, "r") as f:
                manifest_yaml = f.read()

            tool_manifest = await tool_registry.register_manifest_from_yaml(
                manifest_yaml=manifest_yaml,
                source=str(manifest_file)
            )

            print(f"✓ Successfully registered: {tool_manifest.display_name} (v{tool_manifest.version})")
            print(f"  - Operations: {len(tool_manifest.operations)}")

            for op_id, operation in tool_manifest.operations.items():
                print(f"    • {op_id} ({operation.side_effect.value})")

            registered_count += 1

        except Exception as e:
            print(f"✗ Failed to register {manifest_file.name}: {e}")
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"Registration complete:")
    print(f"  ✓ Registered: {registered_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"{'='*60}")


async def list_registered_tools():
    """List all currently registered tools"""

    print("\nCurrently registered tools:")
    print("=" * 60)

    try:
        tools = await tool_registry.list_tools()

        if not tools:
            print("No tools registered yet.")
            return

        for tool in tools:
            print(f"\n{tool.display_name} (v{tool.version})")
            print(f"  Name: {tool.name}")
            print(f"  Operations: {len(tool.operations)}")

            for op_id, operation in tool.operations.items():
                approval = " [REQUIRES APPROVAL]" if operation.requires_approval else ""
                print(f"    • {op_id} - {operation.description}{approval}")

    except Exception as e:
        print(f"Error listing tools: {e}")


async def main():
    """Main registration workflow"""

    print("=" * 60)
    print("Tool Registry - Registration Script")
    print("=" * 60)

    # List existing tools
    await list_registered_tools()

    # Register new tools
    print("\n")
    await register_all_tools()

    # List after registration
    await list_registered_tools()


if __name__ == "__main__":
    asyncio.run(main())
