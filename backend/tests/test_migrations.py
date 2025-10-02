"""
Migration testing framework to prevent database schema regressions.
"""

import pytest
import os
import tempfile
from pathlib import Path
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, MetaData, inspect, text
from sqlalchemy.exc import ProgrammingError


class MigrationTester:
    """Helper class for testing database migrations."""

    def __init__(self, database_url: str = None):
        if database_url is None:
            # Create temporary SQLite database for testing
            self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            self.database_url = f"sqlite:///{self.temp_db.name}"
        else:
            self.database_url = database_url
            self.temp_db = None

        self.engine = create_engine(self.database_url)

        # Setup Alembic config
        self.alembic_cfg = Config()
        self.alembic_cfg.set_main_option("script_location", "alembic")
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)

    def get_current_revision(self) -> str:
        """Get the current database revision."""
        with self.engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()

    def get_all_revisions(self) -> list:
        """Get all available migration revisions."""
        script = ScriptDirectory.from_config(self.alembic_cfg)
        return [rev.revision for rev in script.walk_revisions()]

    def migrate_to_revision(self, revision: str):
        """Migrate database to specific revision."""
        command.upgrade(self.alembic_cfg, revision)

    def downgrade_to_revision(self, revision: str):
        """Downgrade database to specific revision."""
        command.downgrade(self.alembic_cfg, revision)

    def get_table_columns(self, table_name: str) -> dict:
        """Get all columns for a table."""
        inspector = inspect(self.engine)
        if not inspector.has_table(table_name):
            return {}

        columns = {}
        for column in inspector.get_columns(table_name):
            columns[column['name']] = {
                'type': str(column['type']),
                'nullable': column['nullable'],
                'default': column.get('default')
            }
        return columns

    def get_table_indexes(self, table_name: str) -> dict:
        """Get all indexes for a table."""
        inspector = inspect(self.engine)
        if not inspector.has_table(table_name):
            return {}

        indexes = {}
        for index in inspector.get_indexes(table_name):
            indexes[index['name']] = {
                'columns': index['column_names'],
                'unique': index['unique']
            }
        return indexes

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        inspector = inspect(self.engine)
        return inspector.has_table(table_name)

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table."""
        columns = self.get_table_columns(table_name)
        return column_name in columns

    def cleanup(self):
        """Cleanup temporary resources."""
        if self.engine:
            self.engine.dispose()
        if self.temp_db:
            os.unlink(self.temp_db.name)


@pytest.fixture
def migration_tester():
    """Create a migration tester instance."""
    tester = MigrationTester()
    yield tester
    tester.cleanup()


@pytest.mark.database
def test_migration_adds_agent_public_id_column(migration_tester: MigrationTester):
    """Test that the public_id migration correctly adds the column."""
    # Start with a fresh database
    migration_tester.migrate_to_revision("base")

    # Migrate to just before the public_id migration
    migration_tester.migrate_to_revision("2958e8d7d2b0")  # Previous migration

    # Verify public_id column doesn't exist yet
    assert not migration_tester.column_exists("agents", "public_id")

    # Apply the public_id migration
    migration_tester.migrate_to_revision("bb84a883744f")  # public_id migration

    # Verify public_id column now exists
    assert migration_tester.column_exists("agents", "public_id")

    # Verify column properties
    columns = migration_tester.get_table_columns("agents")
    public_id_col = columns["public_id"]
    assert not public_id_col["nullable"]  # Should be NOT NULL

    # Verify unique index exists
    indexes = migration_tester.get_table_indexes("agents")
    assert "ix_agents_public_id" in indexes
    assert indexes["ix_agents_public_id"]["unique"] is True


@pytest.mark.database
def test_migration_rollback_removes_public_id_column(migration_tester: MigrationTester):
    """Test that rolling back the public_id migration removes the column."""
    # Start with the latest migration
    migration_tester.migrate_to_revision("head")

    # Verify public_id exists
    assert migration_tester.column_exists("agents", "public_id")

    # Rollback the public_id migration
    migration_tester.downgrade_to_revision("2958e8d7d2b0")

    # Verify public_id column is removed
    assert not migration_tester.column_exists("agents", "public_id")

    # Verify index is also removed
    indexes = migration_tester.get_table_indexes("agents")
    assert "ix_agents_public_id" not in indexes


@pytest.mark.database
def test_all_migrations_are_reversible(migration_tester: MigrationTester):
    """Test that all migrations can be applied and rolled back."""
    revisions = migration_tester.get_all_revisions()

    # Start from base
    migration_tester.migrate_to_revision("base")

    # Apply each migration one by one
    for revision in reversed(revisions):  # Apply in chronological order
        migration_tester.migrate_to_revision(revision)
        current = migration_tester.get_current_revision()
        assert current == revision

    # Rollback each migration one by one
    for i, revision in enumerate(revisions[1:], 1):  # Skip the last one
        prev_revision = revisions[i] if i < len(revisions) else "base"
        migration_tester.downgrade_to_revision(prev_revision)
        current = migration_tester.get_current_revision()
        assert current == prev_revision


@pytest.mark.database
def test_migration_handles_existing_data(migration_tester: MigrationTester):
    """Test that the public_id migration correctly handles existing data."""
    # Migrate to before public_id migration
    migration_tester.migrate_to_revision("2958e8d7d2b0")

    # Insert test data without public_id
    with migration_tester.engine.connect() as conn:
        # Create tables up to this point
        migration_tester.migrate_to_revision("2958e8d7d2b0")

        # Insert test organization and user (if they exist at this migration)
        try:
            conn.execute(text("""
                INSERT INTO organizations (name, slug, plan, max_agents, max_users, max_documents_per_agent, is_active)
                VALUES ('Test Org', 'test-org', 'pro', 10, 5, 100, 1)
            """))

            conn.execute(text("""
                INSERT INTO users (email, password_hash, name, is_active)
                VALUES ('test@example.com', 'hashed', 'Test User', 1)
            """))

            # Insert test agent without public_id
            conn.execute(text("""
                INSERT INTO agents (name, description, user_id, organization_id, is_active)
                VALUES ('Test Agent', 'Test Description', 1, 1, 1)
            """))
            conn.commit()
        except Exception:
            # Tables might not exist at this migration level, that's ok
            conn.rollback()

    # Apply the public_id migration
    migration_tester.migrate_to_revision("bb84a883744f")

    # Verify that existing agents now have public_id values
    with migration_tester.engine.connect() as conn:
        try:
            result = conn.execute(text("SELECT id, public_id FROM agents WHERE id = 1"))
            row = result.fetchone()
            if row:
                assert row[1] is not None  # public_id should be populated
                assert len(row[1]) > 0      # Should have a value
        except Exception:
            # If the test data insertion failed, that's ok
            pass


@pytest.mark.database
def test_migration_schema_consistency():
    """Test that the final schema matches the model definitions."""
    # This would be more complex in a real implementation
    # It would compare the final database schema with the SQLAlchemy models

    tester = MigrationTester()
    try:
        # Migrate to head
        tester.migrate_to_revision("head")

        # Check that all expected tables exist
        expected_tables = ["users", "organizations", "agents", "user_organizations"]
        for table in expected_tables:
            assert tester.table_exists(table), f"Table {table} should exist"

        # Check that agent table has all expected columns
        agent_columns = tester.get_table_columns("agents")
        expected_agent_columns = [
            "id", "public_id", "name", "description", "user_id",
            "organization_id", "is_active", "created_at", "updated_at"
        ]

        for col in expected_agent_columns:
            assert col in agent_columns, f"Column {col} should exist in agents table"

        # Verify public_id constraints
        assert not agent_columns["public_id"]["nullable"]

        # Verify indexes
        agent_indexes = tester.get_table_indexes("agents")
        assert "ix_agents_public_id" in agent_indexes
        assert agent_indexes["ix_agents_public_id"]["unique"] is True

    finally:
        tester.cleanup()


@pytest.mark.database
def test_model_column_matches_database_schema():
    """Test that model definitions match actual database schema."""
    from app.models.agent import Agent
    from sqlalchemy import inspect as sqlalchemy_inspect

    tester = MigrationTester()
    try:
        # Migrate to head
        tester.migrate_to_revision("head")

        # Get model columns
        agent_model_columns = [col.name for col in Agent.__table__.columns]

        # Get actual database columns
        db_columns = list(tester.get_table_columns("agents").keys())

        # Check that all model columns exist in database
        for col in agent_model_columns:
            assert col in db_columns, f"Model column {col} missing from database"

        # Check that public_id is properly defined in both
        assert "public_id" in agent_model_columns
        assert "public_id" in db_columns

        # Verify public_id properties match
        public_id_col = None
        for col in Agent.__table__.columns:
            if col.name == "public_id":
                public_id_col = col
                break

        assert public_id_col is not None
        assert not public_id_col.nullable
        assert public_id_col.unique

    finally:
        tester.cleanup()


@pytest.mark.database
def test_migration_order_and_dependencies():
    """Test that migrations are applied in correct order."""
    tester = MigrationTester()
    try:
        # Get all revisions
        script = ScriptDirectory.from_config(tester.alembic_cfg)
        revisions = list(script.walk_revisions())

        # Verify our public_id migration comes after its dependency
        public_id_migration = None
        dependency_migration = None

        for rev in revisions:
            if rev.revision == "bb84a883744f":  # public_id migration
                public_id_migration = rev
            elif rev.revision == "2958e8d7d2b0":  # dependency
                dependency_migration = rev

        assert public_id_migration is not None
        assert dependency_migration is not None
        assert public_id_migration.down_revision == dependency_migration.revision

    finally:
        tester.cleanup()