# Test Migration Guide: Updated `create_agent()` Signature

## Overview

The `create_agent()` method signature has been updated to **require** a database session as the first parameter. This enables proper transaction control and atomic operations.

## Breaking Change

**Old Signature:**
```python
async def create_agent(
    self,
    user_id: int,
    organization_id: int,
    name: str,
    description: str,
    system_prompt: str,
    config: Dict[str, Any] = None,
    widget_config: Dict[str, Any] = None
) -> Agent:
```

**New Signature:**
```python
async def create_agent(
    self,
    db: AsyncSession,  # REQUIRED - first parameter
    user_id: int,
    organization_id: int,
    name: str,
    description: str,
    system_prompt: str,
    config: Dict[str, Any] = None,
    widget_config: Dict[str, Any] = None,
    idempotency_key: str = None  # NEW - optional
) -> Agent:
```

## Migration Pattern

### Before:
```python
agent = await db_service.create_agent(
    user_id=test_user.id,
    organization_id=test_organization.id,
    name="Test Agent",
    description="Test",
    system_prompt="Test"
)
```

### After:
```python
async with await get_db_session() as db:
    agent = await db_service.create_agent(
        db=db,
        user_id=test_user.id,
        organization_id=test_organization.id,
        name="Test Agent",
        description="Test",
        system_prompt="Test"
    )
    await db.commit()
```

## Files Requiring Updates

### 1. `backend/tests/test_service_layer.py`

**Affected test methods** (~15 instances):
- `test_agent_creation_ensures_public_id`
- `test_get_agent_by_public_id_works`
- `test_ensure_agent_public_id_handles_missing_id`
- `test_organization_agent_retrieval_with_public_id`
- `test_agent_deletion_cascade_behavior`
- `test_agent_stats_computation_with_valid_data`
- `test_end_to_end_agent_workflow`
- `test_service_error_handling_with_invalid_data`
- `test_service_transaction_rollback_behavior`
- `test_concurrent_agent_creation_safety`
- `test_agent_api_key_generation_uniqueness`
- `test_document_content_hash_validation`
- `test_large_agent_list_retrieval_performance`
- `test_agent_stats_computation_performance`
- `test_agent_document_relationship_consistency`
- `test_user_organization_agent_hierarchy`

**Migration strategy**:
1. Import `get_db_session`: `from app.core.database import get_db_session`
2. Wrap each `create_agent()` call in `async with await get_db_session() as db:`
3. Add `db=db` as first argument
4. Add `await db.commit()` after the call
5. For loops creating multiple agents, keep the session outside the loop and commit once at the end

### 2. `backend/app/services/database_service.py`

**Seed/demo functions** (~2 instances):
- Demo agent creation functions (lines ~1269, ~1278)

**Migration needed:**
```python
# Before
customer_support_agent = await db_service.create_agent(
    user_id=demo_user.id,
    name="Customer Support Bot",
    description="Helps customers",
    system_prompt="You are helpful",
    config={"temperature": 0.7},
    widget_config={"theme": "blue"}
)

# After
async with await get_db_session() as db:
    customer_support_agent = await db_service.create_agent(
        db=db,
        user_id=demo_user.id,
        name="Customer Support Bot",
        description="Helps customers",
        system_prompt="You are helpful",
        config={"temperature": 0.7},
        widget_config={"theme": "blue"}
    )
    await db.commit()
```

## Special Cases

### Creating Multiple Agents in a Loop

**Inefficient** (commits after each):
```python
for i in range(10):
    async with await get_db_session() as db:
        agent = await db_service.create_agent(db=db, ...)
        await db.commit()
```

**Efficient** (single transaction):
```python
async with await get_db_session() as db:
    for i in range(10):
        agent = await db_service.create_agent(db=db, ...)
    await db.commit()  # Commit once at the end
```

### Creating Agent with Error Handling

```python
try:
    async with await get_db_session() as db:
        agent = await db_service.create_agent(db=db, ...)
        await db.commit()
except Exception as e:
    # Session automatically rolls back on exception
    logger.error(f"Failed to create agent: {e}")
```

### Creating Agent in Concurrent Tasks

```python
async def create_agent_task(name: str):
    async with await get_db_session() as db:
        agent = await db_service.create_agent(
            db=db,
            user_id=user_id,
            organization_id=org_id,
            name=name,
            description="Test",
            system_prompt="Test"
        )
        await db.commit()
        return agent

# Run concurrently
agents = await asyncio.gather(*[
    create_agent_task(f"Agent {i}") for i in range(5)
])
```

## Testing Strategy

### 1. Run Individual Test
```bash
cd backend
pytest tests/test_service_layer.py::TestDatabaseServiceRegressionPrevention::test_agent_creation_ensures_public_id -v
```

### 2. Run Full Test Suite
```bash
pytest tests/test_service_layer.py -v
```

### 3. Look for Common Errors

**Missing `db` parameter:**
```
TypeError: create_agent() missing 1 required positional argument: 'db'
```
**Fix:** Add `db=db` as first argument

**Missing commit:**
```
# Agent created but not visible in subsequent queries
```
**Fix:** Add `await db.commit()` after creation

**Syntax errors from indentation:**
```
SyntaxError: invalid syntax
```
**Fix:** Ensure proper indentation inside `async with` block

## Automated Migration Script

Use this script to help migrate test files:

```python
#!/usr/bin/env python3
import re
import sys

def migrate_test_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Add import if not present
    if 'from app.core.database import get_db_session' not in content:
        content = content.replace(
            'from app.models.document import Document',
            'from app.models.document import Document\nfrom app.core.database import get_db_session'
        )

    print(f"Manual migration required for {filepath}")
    print("Please wrap each create_agent() call with:")
    print("  async with await get_db_session() as db:")
    print("      agent = await service.create_agent(db=db, ...)")
    print("      await db.commit()")

if __name__ == '__main__':
    migrate_test_file('tests/test_service_layer.py')
```

## Verification Checklist

- [ ] All `create_agent()` calls have `db` as first argument
- [ ] All calls are wrapped in `async with await get_db_session() as db:`
- [ ] All calls are followed by `await db.commit()`
- [ ] Tests pass: `pytest tests/test_service_layer.py`
- [ ] No syntax errors: `python3 -m black tests/test_service_layer.py --check`

## Benefits of This Change

1. **Atomic Operations**: Agent creation + limit check in single transaction
2. **No Race Conditions**: Concurrent requests can't bypass limits
3. **Explicit Transaction Control**: Caller decides when to commit
4. **Better Performance**: Reuses FastAPI's connection pool
5. **Cleaner API**: One method, one pattern

## Rollback Plan

If migration proves problematic, you can temporarily add a compatibility wrapper:

```python
async def create_agent_legacy(self, user_id: int, organization_id: int, ...):
    """Temporary compatibility wrapper - DEPRECATED"""
    async with await self.get_session() as db:
        agent = await self.create_agent(db, user_id, organization_id, ...)
        await db.commit()
        return agent
```

Then update callers gradually and remove the wrapper once complete.
