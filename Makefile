.PHONY: setup dev build test clean docker-up docker-down migrate

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
