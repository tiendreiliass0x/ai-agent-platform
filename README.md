# AI Agent Platform

A platform for creating True AI agents that can handle users request, computer use by ingest all sort of documents

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

### Setup

1. **Clone and setup environment:**
   ```bash
   make setup
   ```

2. **Edit environment variables:**
   ```bash
   # Edit backend/.env with your API keys
   nano backend/.env
   ```

3. **Start development environment:**
   ```bash
   make dev
   ```

4. **Run initial migration:**
   ```bash
   make migrate-up
   ```

5. **Start Celery worker (for URL discovery/crawl):**
   - Ensure Redis is running. If your Redis container is named `concierge-ai-redis-1`, set the backend to use it:
     - Edit `backend/.env` and set `REDIS_URL=redis://concierge-ai-redis-1:6379`
     - Optionally set `FIRECRAWL_API_KEY=...` to enable Firecrawl-based discovery
   - Start the worker (local):
     ```bash
     make worker
     ```
   - Or via Docker using the backend image:
     ```bash
     make worker-docker
     ```
   - If you already have a Redis container (e.g., `concierge-ai-redis-1`) and want the worker container to use it:
     ```bash
     make worker-host-redis host=redis://concierge-ai-redis-1:6379
     ```

### API Documentation
- FastAPI Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Architecture

```
├── backend/          # FastAPI Python backend
│   ├── app/
│   │   ├── api/      # API endpoints
│   │   ├── core/     # Configuration
│   │   ├── models/   # Database models
│   │   ├── services/ # Business logic
│   │   └── utils/    # Utilities
│   ├── alembic/      # Database migrations
│   └── tests/        # Tests
├── frontend/         # Next.js React frontend (coming soon)
└── shared/           # Shared types/utilities
```

## 🛠️ Development

### Available Commands

```bash
make dev              # Start development server
make docker-up        # Start with Docker
make migrate-create   # Create new migration
make migrate-up       # Run migrations
make test            # Run tests
make lint            # Lint code
make clean           # Clean up Docker
make worker          # Start Celery worker locally
make worker-docker   # Start Celery worker via docker-compose
make worker-host-redis host=redis://concierge-ai-redis-1:6379  # Worker using existing Redis container
make demo-agent-flow # Run end-to-end agent creation demo script
make demo-agent-simple # Run simple agent creation demo
make demo-pdf-ingestion file=/path/to/file.pdf # Run PDF ingestion demo
make demo-pdf-search # Run vector search demo
make demo-security   # Run document security demo
```

### Key Features

- **Document Processing**: Upload PDFs, websites, FAQs
- **Vector Search**: Semantic search with embeddings
- **Real-time Chat**: WebSocket support for live conversations
- **Easy Embedding**: One-line widget integration
- **Analytics**: Conversation tracking and insights

## 🎯 Roadmap

- [x] Backend API structure
- [x] Database models and migrations
- [x] Docker development environment
- [ ] Document processing pipeline
- [ ] RAG implementation
- [ ] Authentication system
- [ ] Frontend dashboard
- [ ] Chat widget
- [ ] Billing integration

## 📝 Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure:

```env
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
DATABASE_URL=postgresql+asyncpg://...
```
