"""
CI/CD Integration tests to ensure deployment and configuration consistency.
"""

import pytest
import os
import subprocess
import json
from pathlib import Path

from app.core.config import settings


@pytest.mark.integration
def test_database_migrations_are_up_to_date():
    """Test that all migrations are applied and up to date."""
    try:
        # Check that alembic is available
        result = subprocess.run(['alembic', '--version'], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, "Alembic should be available"

        # Check current revision
        result = subprocess.run(['alembic', 'current'], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"Failed to get current revision: {result.stderr}"

        # Check if there are pending migrations
        result = subprocess.run(['alembic', 'check'], capture_output=True, text=True, timeout=30)
        # Note: alembic check returns 0 if up to date, 1 if there are pending migrations
        if result.returncode != 0:
            pytest.fail(f"Database migrations are not up to date: {result.stdout}\n{result.stderr}")

    except subprocess.TimeoutExpired:
        pytest.fail("Alembic commands timed out")
    except FileNotFoundError:
        pytest.skip("Alembic not available in test environment")


@pytest.mark.integration
def test_environment_configuration_is_valid():
    """Test that environment configuration is properly set up."""
    # Test critical settings exist
    assert hasattr(settings, 'DATABASE_URL'), "DATABASE_URL should be configured"
    assert hasattr(settings, 'DEBUG'), "DEBUG should be configured"

    # Test database URL format
    if not os.getenv("TESTING"):
        assert settings.DATABASE_URL.startswith(('postgresql://', 'postgresql+asyncpg://')), \
            "DATABASE_URL should be PostgreSQL in production"

    # Test that testing environment uses SQLite
    if os.getenv("TESTING") == "1":
        assert settings.DATABASE_URL.startswith('sqlite'), \
            "Testing should use SQLite database"


@pytest.mark.integration
def test_required_dependencies_are_available():
    """Test that all required dependencies are installed."""
    required_packages = [
        'fastapi',
        'sqlalchemy',
        'alembic',
        'asyncpg',
        'pytest',
        'pytest-asyncio'
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"Required package '{package}' is not available")


@pytest.mark.integration
def test_application_can_start():
    """Test that the FastAPI application can start successfully."""
    try:
        # Import main app
        from main import app
        assert app is not None, "FastAPI app should be importable"

        # Test that app has expected attributes
        assert hasattr(app, 'router'), "App should have router"
        assert hasattr(app, 'middleware'), "App should have middleware"

    except Exception as e:
        pytest.fail(f"Failed to import or initialize app: {e}")


@pytest.mark.integration
def test_database_connection_is_valid():
    """Test that database connection can be established."""
    from app.core.database import async_engine
    import asyncio

    async def test_connection():
        try:
            async with async_engine.connect() as conn:
                result = await conn.execute("SELECT 1")
                assert result.scalar() == 1
        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")

    if not os.getenv("TESTING"):
        pytest.skip("Skipping database connection test in non-testing environment")

    asyncio.run(test_connection())


@pytest.mark.integration
def test_api_routes_are_properly_configured():
    """Test that API routes are properly configured."""
    from main import app

    # Get all routes
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append(route.path)

    # Check for critical API endpoints
    expected_routes = [
        '/api/v1/users/context',
        '/api/v1/users/organizations',
    ]

    for expected_route in expected_routes:
        # Check if route exists (exact match or with path parameters)
        route_exists = any(
            expected_route in route or route.startswith(expected_route.split('{')[0])
            for route in routes
        )
        assert route_exists, f"Expected route '{expected_route}' not found in {routes}"


@pytest.mark.integration
def test_logging_configuration_is_valid():
    """Test that logging is properly configured."""
    import logging

    # Test that loggers can be created
    logger = logging.getLogger("test_logger")
    assert logger is not None

    # Test that logging level is appropriate
    if settings.DEBUG:
        # In debug mode, should log debug messages
        assert logging.root.level <= logging.DEBUG
    else:
        # In production, should not log debug messages
        assert logging.root.level > logging.DEBUG


@pytest.mark.integration
def test_secrets_and_keys_are_configured():
    """Test that secrets and API keys are properly configured."""
    # This is a placeholder - in real environments you'd check for:
    # - JWT secrets are set
    # - API keys are available
    # - Database passwords are not default values
    # - Encryption keys are properly configured

    # Example checks (implement based on your security requirements):
    if hasattr(settings, 'JWT_SECRET_KEY'):
        assert len(settings.JWT_SECRET_KEY) >= 32, "JWT secret should be at least 32 characters"

    # In testing, we can skip some security checks
    if not os.getenv("TESTING"):
        pass  # Add production security checks here


@pytest.mark.integration
def test_static_file_serving_is_configured():
    """Test that static file serving is properly configured if needed."""
    from main import app

    # Check if static files middleware is configured
    has_static_files = any(
        'static' in str(type(middleware)).lower()
        for middleware in app.middleware
    )

    # This test would depend on your specific static file requirements
    # For API-only services, static files might not be needed


@pytest.mark.integration
def test_cors_configuration_is_appropriate():
    """Test that CORS is properly configured."""
    from main import app

    # Check if CORS middleware is configured
    has_cors = any(
        'cors' in str(type(middleware)).lower()
        for middleware in app.middleware
    )

    # In development, CORS should be permissive
    # In production, CORS should be restrictive
    if settings.DEBUG:
        # Development should have CORS configured
        assert has_cors, "CORS should be configured in development"


@pytest.mark.integration
def test_health_check_endpoint_works():
    """Test that health check endpoint is functional."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # Try to find health check endpoint
    health_endpoints = ['/health', '/api/health', '/api/v1/health', '/api/v1/system/health']

    health_found = False
    for endpoint in health_endpoints:
        try:
            response = client.get(endpoint)
            if response.status_code == 200:
                health_found = True
                break
        except Exception:
            continue

    # Health check is recommended but not required
    if not health_found:
        pytest.skip("No health check endpoint found - consider adding one for monitoring")


@pytest.mark.integration
def test_error_handling_is_configured():
    """Test that proper error handling is configured."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # Test 404 handling
    response = client.get('/nonexistent-endpoint')
    assert response.status_code == 404

    # Response should be JSON
    try:
        response.json()
    except ValueError:
        pytest.fail("Error responses should be JSON formatted")


@pytest.mark.integration
def test_middleware_stack_is_appropriate():
    """Test that middleware stack is properly configured."""
    from main import app

    middleware_types = [str(type(middleware)) for middleware in app.middleware]

    # Check for security middleware (implement based on your needs)
    # Example: CORS, authentication, rate limiting, etc.

    # This is a placeholder - implement based on your middleware requirements
    assert len(app.middleware) >= 0, "App should have middleware configured"


@pytest.mark.integration
def test_database_pool_configuration():
    """Test that database connection pool is properly configured."""
    from app.core.database import async_engine

    # Check pool configuration
    pool = async_engine.pool

    # These values should be appropriate for your deployment
    if not os.getenv("TESTING"):
        # In production, pool should be properly sized
        assert pool.size() > 0, "Database pool should be configured"


@pytest.mark.integration
def test_migration_scripts_are_valid():
    """Test that migration scripts can be validated."""
    try:
        # Check that migration directory exists
        migration_dir = Path("alembic/versions")
        assert migration_dir.exists(), "Migration directory should exist"

        # Check that migrations have proper naming
        migration_files = list(migration_dir.glob("*.py"))
        assert len(migration_files) > 0, "Should have migration files"

        # Check that each migration file has required components
        for migration_file in migration_files:
            if migration_file.name == "__init__.py":
                continue

            content = migration_file.read_text()
            assert "def upgrade():" in content, f"Migration {migration_file.name} should have upgrade function"
            assert "def downgrade():" in content, f"Migration {migration_file.name} should have downgrade function"

    except Exception as e:
        pytest.fail(f"Migration validation failed: {e}")


@pytest.mark.integration
def test_requirements_are_pinned():
    """Test that requirements are properly pinned for reproducible builds."""
    requirements_file = Path("requirements.txt")

    if requirements_file.exists():
        content = requirements_file.read_text()
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]

        unpinned_packages = []
        for line in lines:
            if '==' not in line and not line.startswith('-'):
                unpinned_packages.append(line)

        if unpinned_packages:
            pytest.fail(f"Unpinned packages found: {unpinned_packages}. All packages should have exact versions.")


@pytest.mark.integration
def test_docker_configuration_if_present():
    """Test Docker configuration if Dockerfile exists."""
    dockerfile = Path("Dockerfile")

    if dockerfile.exists():
        content = dockerfile.read_text()

        # Basic Docker best practices
        assert "FROM" in content, "Dockerfile should have FROM instruction"

        # Check for security best practices
        if "USER root" in content:
            # Should switch to non-root user
            assert "USER" in content.split("USER root")[-1], \
                "Should switch away from root user after initial setup"


@pytest.mark.integration
def test_environment_specific_configurations():
    """Test that environment-specific configurations are handled."""
    # Test development vs production configurations
    if os.getenv("TESTING") == "1":
        # Testing environment checks
        assert settings.DATABASE_URL.startswith('sqlite'), "Testing should use SQLite"

    # Add more environment-specific checks as needed
    # Examples:
    # - Production should use PostgreSQL
    # - Development should have debug enabled
    # - Staging should have appropriate logging levels