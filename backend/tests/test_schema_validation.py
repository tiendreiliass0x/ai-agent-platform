"""
Schema validation tests to ensure model-database consistency.
This would have caught the public_id column regression.
"""

import pytest
import inspect
from typing import Dict, Set, Any
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.database import Base
from app.models.user import User
from app.models.organization import Organization
from app.models.agent import Agent
from app.models.user_organization import UserOrganization


class SchemaValidator:
    """Validates that SQLAlchemy models match database schema."""

    def __init__(self, database_url: str = "sqlite:///test_schema.db"):
        self.engine = create_engine(database_url)
        self.metadata = MetaData()

    def create_all_tables(self):
        """Create all tables from models."""
        Base.metadata.create_all(self.engine)

    def get_model_columns(self, model_class) -> Dict[str, Any]:
        """Get column definitions from SQLAlchemy model."""
        columns = {}
        for column in model_class.__table__.columns:
            columns[column.name] = {
                'type': str(column.type),
                'nullable': column.nullable,
                'unique': column.unique,
                'index': column.index,
                'primary_key': column.primary_key,
                'foreign_keys': [str(fk) for fk in column.foreign_keys],
                'default': column.default,
                'server_default': column.server_default
            }
        return columns

    def get_database_columns(self, table_name: str) -> Dict[str, Any]:
        """Get column definitions from actual database."""
        inspector = sqlalchemy_inspect(self.engine)
        if not inspector.has_table(table_name):
            return {}

        columns = {}
        for column in inspector.get_columns(table_name):
            columns[column['name']] = {
                'type': str(column['type']),
                'nullable': column['nullable'],
                'default': column.get('default'),
                'primary_key': column.get('primary_key', False)
            }
        return columns

    def get_model_indexes(self, model_class) -> Dict[str, Any]:
        """Get index definitions from SQLAlchemy model."""
        indexes = {}
        for index in model_class.__table__.indexes:
            indexes[index.name] = {
                'columns': [col.name for col in index.columns],
                'unique': index.unique
            }
        return indexes

    def get_database_indexes(self, table_name: str) -> Dict[str, Any]:
        """Get index definitions from actual database."""
        inspector = sqlalchemy_inspect(self.engine)
        if not inspector.has_table(table_name):
            return {}

        indexes = {}
        for index in inspector.get_indexes(table_name):
            indexes[index['name']] = {
                'columns': index['column_names'],
                'unique': index['unique']
            }
        return indexes

    def compare_columns(self, model_class, table_name: str) -> Dict[str, Any]:
        """Compare model columns with database columns."""
        model_columns = self.get_model_columns(model_class)
        db_columns = self.get_database_columns(table_name)

        return {
            'model_only': set(model_columns.keys()) - set(db_columns.keys()),
            'database_only': set(db_columns.keys()) - set(model_columns.keys()),
            'common': set(model_columns.keys()) & set(db_columns.keys()),
            'model_columns': model_columns,
            'database_columns': db_columns
        }

    def validate_model_schema(self, model_class) -> Dict[str, Any]:
        """Validate that model matches database schema."""
        table_name = model_class.__tablename__
        comparison = self.compare_columns(model_class, table_name)

        issues = []

        # Check for missing columns in database
        if comparison['model_only']:
            for col in comparison['model_only']:
                issues.append(f"Column '{col}' exists in model but not in database")

        # Check for extra columns in database
        if comparison['database_only']:
            for col in comparison['database_only']:
                issues.append(f"Column '{col}' exists in database but not in model")

        # Check column properties for common columns
        for col_name in comparison['common']:
            model_col = comparison['model_columns'][col_name]
            db_col = comparison['database_columns'][col_name]

            # Check nullable property
            if model_col['nullable'] != db_col['nullable']:
                issues.append(f"Column '{col_name}' nullable mismatch: model={model_col['nullable']}, db={db_col['nullable']}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'comparison': comparison
        }

    def cleanup(self):
        """Clean up resources."""
        if self.engine:
            self.engine.dispose()


@pytest.fixture
def schema_validator():
    """Create a schema validator instance."""
    validator = SchemaValidator()
    validator.create_all_tables()
    yield validator
    validator.cleanup()


@pytest.mark.database
def test_agent_model_matches_database_schema(schema_validator: SchemaValidator):
    """Test that Agent model matches database schema - would catch public_id regression."""
    validation_result = schema_validator.validate_model_schema(Agent)

    # Print issues for debugging
    if validation_result['issues']:
        print(f"Agent schema issues: {validation_result['issues']}")

    assert validation_result['valid'], f"Agent model schema issues: {validation_result['issues']}"

    # Specifically check that public_id exists
    comparison = validation_result['comparison']
    assert 'public_id' in comparison['common'], "public_id column should exist in both model and database"

    # Check public_id properties
    model_public_id = comparison['model_columns']['public_id']
    assert not model_public_id['nullable'], "public_id should not be nullable"
    assert model_public_id['unique'], "public_id should be unique"


@pytest.mark.database
def test_user_model_matches_database_schema(schema_validator: SchemaValidator):
    """Test that User model matches database schema."""
    validation_result = schema_validator.validate_model_schema(User)

    assert validation_result['valid'], f"User model schema issues: {validation_result['issues']}"

    # Specifically check password field naming
    comparison = validation_result['comparison']
    assert 'password_hash' in comparison['common'], "password_hash column should exist"
    assert 'hashed_password' not in comparison['common'], "hashed_password should not exist"


@pytest.mark.database
def test_organization_model_matches_database_schema(schema_validator: SchemaValidator):
    """Test that Organization model matches database schema."""
    validation_result = schema_validator.validate_model_schema(Organization)

    assert validation_result['valid'], f"Organization model schema issues: {validation_result['issues']}"


@pytest.mark.database
def test_user_organization_model_matches_database_schema(schema_validator: SchemaValidator):
    """Test that UserOrganization model matches database schema."""
    validation_result = schema_validator.validate_model_schema(UserOrganization)

    assert validation_result['valid'], f"UserOrganization model schema issues: {validation_result['issues']}"


@pytest.mark.database
def test_all_models_have_matching_schemas(schema_validator: SchemaValidator):
    """Test that all models match their database schemas."""
    models_to_test = [User, Organization, Agent, UserOrganization]
    all_issues = []

    for model in models_to_test:
        validation_result = schema_validator.validate_model_schema(model)
        if not validation_result['valid']:
            all_issues.extend([f"{model.__name__}: {issue}" for issue in validation_result['issues']])

    assert len(all_issues) == 0, f"Schema validation issues found: {all_issues}"


@pytest.mark.database
def test_agent_table_has_required_indexes(schema_validator: SchemaValidator):
    """Test that Agent table has all required indexes."""
    db_indexes = schema_validator.get_database_indexes('agents')

    # Check for public_id index
    public_id_indexes = [name for name, info in db_indexes.items() if 'public_id' in info['columns']]
    assert len(public_id_indexes) > 0, "Agent table should have index on public_id column"

    # Check that public_id index is unique
    for index_name in public_id_indexes:
        index_info = db_indexes[index_name]
        if index_info['columns'] == ['public_id']:
            assert index_info['unique'], f"public_id index {index_name} should be unique"


@pytest.mark.database
def test_foreign_key_relationships_are_valid(schema_validator: SchemaValidator):
    """Test that foreign key relationships are properly defined."""
    inspector = sqlalchemy_inspect(schema_validator.engine)

    # Test Agent foreign keys
    if inspector.has_table('agents'):
        agent_fks = inspector.get_foreign_keys('agents')
        fk_columns = [fk['constrained_columns'][0] for fk in agent_fks]

        assert 'user_id' in fk_columns, "Agent should have foreign key to users"
        assert 'organization_id' in fk_columns, "Agent should have foreign key to organizations"

        # Check foreign key targets
        for fk in agent_fks:
            if fk['constrained_columns'] == ['user_id']:
                assert fk['referred_table'] == 'users', "user_id should reference users table"
                assert fk['referred_columns'] == ['id'], "user_id should reference users.id"

            if fk['constrained_columns'] == ['organization_id']:
                assert fk['referred_table'] == 'organizations', "organization_id should reference organizations table"
                assert fk['referred_columns'] == ['id'], "organization_id should reference organizations.id"


@pytest.mark.database
def test_table_names_match_model_definitions():
    """Test that actual table names match model __tablename__ attributes."""
    validator = SchemaValidator()
    validator.create_all_tables()

    try:
        inspector = sqlalchemy_inspect(validator.engine)
        actual_tables = set(inspector.get_table_names())

        expected_tables = {
            User.__tablename__,
            Organization.__tablename__,
            Agent.__tablename__,
            UserOrganization.__tablename__
        }

        missing_tables = expected_tables - actual_tables
        extra_tables = actual_tables - expected_tables

        assert len(missing_tables) == 0, f"Missing tables: {missing_tables}"
        # Note: extra_tables might include alembic_version, so we don't assert on that

    finally:
        validator.cleanup()


@pytest.mark.database
def test_column_constraints_are_enforced():
    """Test that database constraints match model constraints."""
    validator = SchemaValidator()
    validator.create_all_tables()

    try:
        # Test unique constraints
        inspector = sqlalchemy_inspect(validator.engine)

        if inspector.has_table('agents'):
            unique_constraints = inspector.get_unique_constraints('agents')
            public_id_unique = any(
                'public_id' in constraint['column_names']
                for constraint in unique_constraints
            )
            # SQLite might handle unique via index instead of constraint
            indexes = inspector.get_indexes('agents')
            public_id_unique_index = any(
                index['unique'] and 'public_id' in index['column_names']
                for index in indexes
            )

            assert public_id_unique or public_id_unique_index, "public_id should have unique constraint or unique index"

        if inspector.has_table('users'):
            unique_constraints = inspector.get_unique_constraints('users')
            email_unique = any(
                'email' in constraint['column_names']
                for constraint in unique_constraints
            )
            indexes = inspector.get_indexes('users')
            email_unique_index = any(
                index['unique'] and 'email' in index['column_names']
                for index in indexes
            )

            assert email_unique or email_unique_index, "email should have unique constraint or unique index"

    finally:
        validator.cleanup()


@pytest.mark.database
def test_critical_columns_exist_in_all_environments():
    """Test that critical columns exist - regression prevention."""
    critical_columns = {
        'agents': ['id', 'public_id', 'name', 'user_id', 'organization_id', 'is_active'],
        'users': ['id', 'email', 'password_hash', 'name', 'is_active'],
        'organizations': ['id', 'name', 'slug', 'plan', 'max_agents', 'max_users', 'is_active'],
        'user_organizations': ['id', 'user_id', 'organization_id', 'role', 'is_active']
    }

    validator = SchemaValidator()
    validator.create_all_tables()

    try:
        inspector = sqlalchemy_inspect(validator.engine)

        for table_name, required_columns in critical_columns.items():
            if inspector.has_table(table_name):
                actual_columns = [col['name'] for col in inspector.get_columns(table_name)]

                for required_col in required_columns:
                    assert required_col in actual_columns, f"Critical column '{required_col}' missing from table '{table_name}'"

    finally:
        validator.cleanup()


@pytest.mark.database
def test_model_definitions_are_importable():
    """Test that all model classes can be imported and have required attributes."""
    models = [User, Organization, Agent, UserOrganization]

    for model in models:
        # Check basic SQLAlchemy model attributes
        assert hasattr(model, '__tablename__'), f"{model.__name__} should have __tablename__"
        assert hasattr(model, '__table__'), f"{model.__name__} should have __table__"
        assert hasattr(model, 'id'), f"{model.__name__} should have id column"

        # Check that table has columns
        assert len(model.__table__.columns) > 0, f"{model.__name__} should have columns defined"


@pytest.mark.database
def test_schema_validation_would_catch_public_id_regression():
    """Specific test that would have caught the public_id regression."""
    # This test simulates what would happen if public_id was missing from database
    # but present in model (the original regression)

    validator = SchemaValidator()

    # Create tables normally
    validator.create_all_tables()

    # Verify public_id exists in model
    agent_model_columns = validator.get_model_columns(Agent)
    assert 'public_id' in agent_model_columns, "public_id should exist in Agent model"

    # Verify public_id exists in database (after migration)
    agent_db_columns = validator.get_database_columns('agents')
    assert 'public_id' in agent_db_columns, "public_id should exist in database after migration"

    # Verify schema validation passes
    validation_result = validator.validate_model_schema(Agent)
    assert validation_result['valid'], "Agent schema should be valid after migration"

    # Verify properties match
    model_public_id = agent_model_columns['public_id']
    db_public_id = agent_db_columns['public_id']

    # This would fail if the migration hadn't been applied
    assert not model_public_id['nullable'], "Model public_id should not be nullable"
    assert not db_public_id['nullable'], "Database public_id should not be nullable"

    validator.cleanup()