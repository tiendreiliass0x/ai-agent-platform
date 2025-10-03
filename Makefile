.PHONY: setup dev build test clean docker-up docker-down migrate worker worker-docker worker-host-redis demo-agent-flow demo-agent-simple demo-pdf-ingestion demo-pdf-search demo-security

# Setup development environment
setup:
	cp backend/.env.example backend/.env
	@echo "Please edit backend/.env with your API keys and settings"

# Start development environment
dev:
	docker-compose up postgres redis -d
	sleep 5
	cd backend && python -m uvicorn main:app --reload

# Start with Docker
docker-up:
	docker-compose up -d

# Stop Docker services
docker-down:
	docker-compose down

# Database migration commands
migrate-create:
	cd backend && alembic revision --autogenerate -m "$(msg)"

migrate-up:
	cd backend && alembic upgrade head

migrate-down:
	cd backend && alembic downgrade -1

# Install dependencies
install:
	cd backend && pip install -r requirements.txt
	npm install

# Run tests
test:
	cd backend && pytest
	npm test

# Lint code
lint:
	cd backend && ruff check .
	npm run lint

# Security audit
security-check:
	cd backend && bandit -q -r app -x app/tests,app/__pycache__
	@echo "Bandit security scan completed"

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Build for production
build:
	docker-compose build
	npm run build

# Start Celery worker (local)
worker:
	cd backend && celery -A app.celery_app.celery_app worker -l info

# Start Celery worker in Docker (uses backend image and REDIS_URL)
worker-docker:
	docker-compose run --rm backend celery -A app.celery_app.celery_app worker -l info

# Start Celery worker in Docker using external Redis host (override REDIS_URL)
worker-host-redis:
	@if [ -z "$(host)" ]; then echo "Usage: make worker-host-redis host=redis://concierge-ai-redis-1:6379"; exit 1; fi
	docker-compose run --rm -e REDIS_URL=$(host) backend celery -A app.celery_app.celery_app worker -l info

# Demo scripts (manual E2E/dev verification)
demo-agent-flow:
	cd backend && python scripts/agent_creation_flow_demo.py

demo-agent-simple:
	cd backend && python scripts/agent_creation_simple_demo.py

demo-pdf-ingestion:
	@if [ -z "$(file)" ]; then echo "Usage: make demo-pdf-ingestion file=/path/to/file.pdf"; exit 1; fi
	cd backend && python scripts/pdf_ingestion_demo.py --file "$(file)"

demo-pdf-search:
	cd backend && python scripts/pdf_search_demo.py

demo-security:
	cd backend && python scripts/security_system_demo.py
