"""
Global pytest configuration and fixtures for comprehensive testing.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set test environment
os.environ["TESTING"] = "1"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create a test database engine."""
    # Import database models here to avoid import issues
    from app.core.database import Base

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine, monkeypatch) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session and override the global session factory."""
    from app.core.database import Base
    import app.core.database

    # Create a new session for each test
    async_session = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Clean up tables before each test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Override the global AsyncSessionLocal to use our test session factory
    monkeypatch.setattr(app.core.database, 'AsyncSessionLocal', async_session)

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession):
    """Create a test user."""
    from app.models.user import User

    user = User(
        email="test@example.com",
        name="Test User",
        password_hash="hashed_password_here",
        is_active=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_organization(db_session: AsyncSession):
    """Create a test organization."""
    from app.models.organization import Organization

    org = Organization(
        name="Test Organization",
        slug="test-org",
        plan="pro",
        max_agents=10,
        max_users=5,
        max_documents_per_agent=100,
        is_active=True
    )
    db_session.add(org)
    await db_session.commit()
    await db_session.refresh(org)
    return org


@pytest_asyncio.fixture
async def test_agent(db_session: AsyncSession, test_organization):
    """Create a test agent."""
    from app.models.agent import Agent

    agent = Agent(
        name="Test Agent",
        description="A test agent",
        personality="helpful",
        organization_id=test_organization.id,
        is_active=True,
        public_id="test-agent-123"
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest.fixture
def client():
    """Create a test client."""
    from main import app
    return TestClient(app)


@pytest.fixture
def authenticated_client(test_user):
    """Create an authenticated test client."""
    from main import app
    from app.core.auth import get_current_user

    def mock_get_current_user():
        return test_user

    app.dependency_overrides[get_current_user] = mock_get_current_user
    client = TestClient(app)

    yield client

    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
def mock_db_session(db_session: AsyncSession):
    """Override database dependency for testing."""
    from main import app
    from app.core.database import get_db

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    yield db_session

    # Clean up
    app.dependency_overrides.clear()


# Test data fixtures
@pytest.fixture
def sample_documents():
    """Sample document data for testing."""
    return [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "ml_basics.pdf", "page": 1},
            "score": 0.8
        },
        {
            "text": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"source": "dl_guide.pdf", "page": 3},
            "score": 0.7
        },
        {
            "text": "Natural language processing involves text analysis.",
            "metadata": {"source": "nlp_intro.pdf", "page": 1},
            "score": 0.6
        }
    ]


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn without explicit programming."},
        {"role": "user", "content": "How does it work?"},
    ]


# Utility functions for tests
def create_mock_response(status_code: int = 200, json_data: dict = None):
    """Create a mock HTTP response."""
    class MockResponse:
        def __init__(self, status_code: int, json_data: dict = None):
            self.status_code = status_code
            self._json_data = json_data or {}

        def json(self):
            return self._json_data

        @property
        def text(self):
            return str(self._json_data)

    return MockResponse(status_code, json_data)


# Async test utilities
async def async_return(value):
    """Helper for mocking async functions."""
    return value


# Database utilities
async def create_test_data(db_session: AsyncSession):
    """Create comprehensive test data."""
    from app.models.organization import Organization
    from app.models.user import User
    from app.models.agent import Agent

    # Create organization
    org = Organization(
        name="Test Corp",
        slug="test-corp",
        plan="enterprise",
        max_agents=100,
        max_users=50,
        max_documents_per_agent=1000,
        is_active=True
    )
    db_session.add(org)
    await db_session.flush()

    # Create user
    user = User(
        email="admin@testcorp.com",
        name="Admin User",
        password_hash="hashed",
        is_active=True
    )
    db_session.add(user)
    await db_session.flush()

    # Create agents
    agents = []
    for i in range(3):
        agent = Agent(
            name=f"Agent {i+1}",
            description=f"Test agent number {i+1}",
            personality="professional",
            organization_id=org.id,
            is_active=True,
            public_id=f"agent-{i+1}-test"
        )
        db_session.add(agent)
        agents.append(agent)

    await db_session.commit()

    return {
        "organization": org,
        "user": user,
        "agents": agents
    }


# Markers for different test types
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.api = pytest.mark.api
pytest.mark.database = pytest.mark.database
pytest.mark.slow = pytest.mark.slow