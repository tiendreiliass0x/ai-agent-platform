# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An AI agent platform for creating intelligent chatbot concierges that companies can embed on their websites. The platform uses RAG (Retrieval-Augmented Generation) to provide context-aware responses based on uploaded documents, websites, and knowledge bases.

## Development Commands

### Environment Setup
```bash
make setup              # Copy .env.example to .env (edit with API keys)
make install            # Install Python and Node dependencies
```

### Running the Application
```bash
make dev                # Start backend with local postgres/redis (docker)
make docker-up          # Start all services with Docker
uvicorn main:app --reload  # Run backend directly (from backend/ dir)
```

### Database Operations
```bash
# From backend/ directory:
alembic revision --autogenerate -m "description"  # Create migration
alembic upgrade head    # Apply migrations
alembic downgrade -1    # Rollback one migration

# Using Makefile:
make migrate-create msg="description"
make migrate-up
make migrate-down
```

### Testing
```bash
make test               # Run all tests
cd backend && pytest    # Run backend tests
pytest tests/test_specific.py  # Run specific test file
pytest -m unit          # Run unit tests only
pytest -m integration   # Run integration tests only
pytest -k "test_name"   # Run tests matching pattern
```

### Code Quality
```bash
make lint               # Lint Python and JS code
cd backend && ruff check .       # Lint backend
make security-check     # Run bandit security scan
```

### Docker Management
```bash
docker-compose up postgres redis -d  # Start dependencies only
docker-compose logs -f backend       # View backend logs
make docker-down        # Stop services
make clean              # Clean up Docker volumes
```

## Architecture

### Tech Stack
- **Backend**: FastAPI (Python 3.11+) with async/await
- **Database**: PostgreSQL with pgvector extension for vector storage
- **Cache/Queue**: Redis for rate limiting and background tasks
- **Vector Search**: Pinecone (primary) with abstraction layer for multiple backends
- **LLMs**: OpenAI GPT-4 and Google Gemini
- **Embeddings**: sentence-transformers (local) and OpenAI embeddings
- **Document Processing**: pymupdf4llm, unstructured, pypdf, trafilatura
- **Web Scraping**: FireCrawl API for website ingestion

### Core System Architecture

**Multi-Tenant Organization Model**: Users belong to organizations, agents belong to organizations. Authorization follows organization boundaries.

**RAG Pipeline**:
1. Document ingestion → `DocumentProcessor` (backend/app/services/document_processor.py)
2. Text chunking with semantic splitting → `SimpleTextSplitter` or `SemanticTextSplitter`
3. Embedding generation → `EmbeddingService` (backend/app/services/embedding_service.py)
4. Vector storage → `VectorStoreService` with provider abstraction (backend/app/services/vector_store_base.py)
5. Query → Retrieval → Reranking → LLM generation → `RAGService` (backend/app/services/rag_service.py)

**Agent Tiers** (backend/app/models/agent.py:8-12):
- `basic`: Standard knowledge base only
- `professional`: + Domain expertise + web search
- `enterprise`: + Custom training + advanced analytics

**Domain Expertise System**: Agents can be configured with:
- **Personas**: Role-based personalities (sales rep, solution engineer, support expert, domain specialist, product expert)
- **Knowledge Packs**: Curated domain-specific information with freshness policies
- **Tool Policies**: Web search, site-specific search, code execution permissions
- **Grounding Modes**: "strict" (only use provided context) or "blended" (allow general knowledge)

### Key Services

**Document Processing** (backend/app/services/document_processor.py):
- Supports PDF, Word, TXT, web scraping
- Uses PyMuPDF for PDFs with markdown conversion
- FireCrawl integration for website crawling
- Semantic chunking with overlap for better retrieval
- Deduplication using SimHash
- Keyword extraction with YAKE

**RAG Service** (backend/app/services/rag_service.py):
- Vector search with configurable top_k
- Reranking with cross-encoder models
- Context compression for token efficiency
- Personality injection for agent responses
- Streaming support via Server-Sent Events

**Personality Service** (backend/app/services/personality_service.py):
- Injects persona characteristics into system prompts
- Enhances responses with personality traits
- Supports built-in persona templates

**Memory & Context** (backend/app/services/memory_service.py):
- Conversation history management
- Customer profile tracking
- Context-aware response generation

**Security System** (backend/app/services/document_security.py):
- File type validation with libmagic
- Content scanning for malicious patterns
- File size limits and upload restrictions

### API Structure

All routes are under `/api/v1/` (backend/app/api/v1/):
- `auth.py`: JWT-based authentication, login, registration
- `agents.py`: Agent CRUD operations, configuration
- `documents.py`: Document upload, processing, management
- `conversations.py`: Conversation history, message tracking
- `domain_expertise.py`: Persona and knowledge pack management
- `chat.py` (endpoints): Public chat API for widget integration
- `websocket_chat.py`: Real-time WebSocket chat

### Database Models

Key models in backend/app/models/:
- `User`: Authentication, belongs to organizations
- `Organization`: Multi-tenant container
- `UserOrganization`: Many-to-many with role permissions
- `Agent`: Chatbot configuration with tier, persona, knowledge packs
- `Document`: Uploaded files/websites with processing status
- `Conversation`: Chat session tracking
- `Message`: Individual messages with role (user/assistant/system)
- `Persona`: Domain expertise personalities with tactics and communication styles
- `KnowledgePack`: Curated domain knowledge with freshness policies
- `CustomerProfile`: User behavior tracking and preferences
- `Escalation`: Human handoff tracking

### Security Features

**Middleware Stack** (backend/main.py):
1. SecurityHeadersMiddleware: CSP, X-Frame-Options, HSTS
2. RateLimitMiddleware: Redis-based rate limiting with configurable windows
3. TrustedHostMiddleware: Host validation in production
4. GZipMiddleware: Response compression
5. CORSMiddleware: Configurable origins from settings

**Authentication** (backend/app/core/auth.py):
- JWT tokens with HS256 algorithm
- Password hashing with bcrypt (12 rounds)
- Token expiration: 24 hours default
- Protected routes require `Authorization: Bearer <token>` header

**Authorization** (backend/app/core/auth.py):
- Organization-based access control
- User roles: owner, admin, member
- Agent-level permissions via organization membership

### Configuration

**Environment Variables** (backend/app/core/config.py):
- `ENVIRONMENT`: "development", "production", "test"
- `DATABASE_URL`: PostgreSQL connection string (uses asyncpg)
- `REDIS_URL`: Redis connection for rate limiting
- `OPENAI_API_KEY`: For GPT models and embeddings
- `GEMINI_API_KEY`: For Gemini models (default LLM)
- `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME`
- `FIRECRAWL_API_KEY`: For web scraping
- `LANGEXTRACT_API_KEY`: For language extraction
- `SECRET_KEY`, `JWT_SECRET_KEY`: Must be set in production (auto-generated in dev)
- `ALLOWED_ORIGINS`: CORS configuration (comma-separated)
- `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW`: Rate limiting config

### Widget Integration

Frontend widget files in `widget/`:
- `agent-widget.js`: Standalone embeddable chat widget
- `agent-widget-complete.js`: Full-featured version with additional UI

Widget connects to `/api/v1/chat/{agent_public_id}` using agent's API key.

## Development Patterns

### Async/Await Usage
- All database operations use SQLAlchemy async sessions (`AsyncSession`)
- Service methods are async and should be awaited
- Never mix sync/async patterns (causes memory leaks)

### Error Handling
- Use structured logging via `LoggerMixin` instead of print statements
- Catch specific exceptions rather than broad `Exception` catches
- Return proper HTTP status codes with meaningful error messages

### Database Sessions
- Import: `from sqlalchemy.ext.asyncio import AsyncSession`
- Use dependency injection: `db: AsyncSession = Depends(get_db)`
- Always use async context managers for transactions

### Testing Markers
Available pytest markers (backend/pytest.ini):
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.api`: API endpoint tests
- `@pytest.mark.database`: Database tests
- `@pytest.mark.slow`: Slow running tests

### Vector Store Abstraction
The system supports multiple vector backends via `VectorStoreInterface` (backend/app/services/vector_store_base.py):
- Primary: Pinecone
- Alternatives: pgvector, Milvus, Redis, Qdrant, Chroma
- Switch by implementing the interface and updating configuration

## Common Tasks

### Adding a New Migration
```bash
cd backend
alembic revision --autogenerate -m "add new field to agents"
# Review the generated migration in alembic/versions/
alembic upgrade head
```

### Adding a New API Endpoint
1. Create/edit router in `backend/app/api/v1/`
2. Add authentication with `current_user: User = Depends(get_current_user)`
3. Add authorization checks for organization access
4. Register router in `backend/app/api/v1/__init__.py`

### Processing a New Document Type
1. Add format detection in `DocumentProcessor.process_file()`
2. Implement extraction logic (see PDF/website examples)
3. Use semantic splitter for better chunking
4. Store vectors with proper metadata

### Adding a New Persona
1. Define in `backend/app/services/persona_templates.py`
2. Include tactics, communication style, response patterns
3. Seed via migration or admin API

## API Documentation

When backend is running:
- Swagger UI: http://localhost:8000/docs (dev only)
- ReDoc: http://localhost:8000/redoc (dev only)
- Health check: http://localhost:8000/health

## Database

PostgreSQL with extensions:
- `uuid-ossp`: UUID generation
- `pg_trgm`: Full-text search support
- `vector` (pgvector): Vector similarity search

Connection runs on port 5432 (docker) or as configured in DATABASE_URL.

## Important Notes

- The frontend directory exists but is not yet implemented (coming soon)
- Default LLM is Gemini (not OpenAI) - see backend/app/services/gemini_service.py
- Rate limiting uses Redis - ensure Redis is running
- Document processing is intentionally NOT using LangChain (removed for lightweight alternatives)
- Agent public IDs are UUIDs used for widget embedding (different from internal integer IDs)
- Security scanning requires libmagic installed (included in Docker image)
