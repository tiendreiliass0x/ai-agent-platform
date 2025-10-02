# Docker Setup for AI Agent Platform Backend

This document explains how to run the AI Agent Platform backend using Docker.

## Prerequisites

- Docker and Docker Compose installed
- Make sure port 5432 (PostgreSQL) and 6379 (Redis) are available on your host machine

## Quick Start

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f backend
   ```

3. **Stop services:**
   ```bash
   docker-compose down
   ```

## Services

The Docker setup includes:

- **PostgreSQL** (port 5432): Main database
- **Redis** (port 6379): Caching and background tasks
- **Backend API** (port 8000): FastAPI application

## Database

PostgreSQL runs on the standard port **5432** and includes:
- Database: `ai_agent_platform`
- User: `postgres`
- Password: `password`

The database is initialized with useful extensions:
- `uuid-ossp`: UUID generation
- `pg_trgm`: Full-text search support

## Environment Configuration

The Docker setup uses environment variables defined in the `docker-compose.yml`. For production, create a `.env` file or use the provided `.env.docker` template.

Key configurations:
- `DATABASE_URL`: Uses Docker internal networking (`postgres:5432`)
- `REDIS_URL`: Uses Docker internal networking (`redis:6379`)
- API keys and secrets should be configured in your environment

## Development

For development with hot reloading:
```bash
docker-compose up
```

The backend container mounts your local directory, so code changes will automatically reload the server.

## Data Persistence

- PostgreSQL data: `postgres_data` volume
- Redis data: `redis_data` volume
- Upload files: Mounted to `./uploads` directory

## Security Features

The Docker image includes:
- `libmagic` for file type detection (security system)
- All necessary dependencies for the document security features

## Troubleshooting

1. **Port conflicts**: If ports 5432 or 6379 are in use, modify the port mappings in `docker-compose.yml`

2. **Database connection issues**: Ensure the database service is healthy before the backend starts (health checks are configured)

3. **Permission issues**: On Linux, you may need to set proper permissions for the uploads directory:
   ```bash
   chmod 755 uploads
   ```

## Production Deployment

For production:
1. Change default passwords in `docker-compose.yml`
2. Use proper environment variables for API keys
3. Configure proper secrets management
4. Set up SSL/TLS termination
5. Consider using Docker Swarm or Kubernetes for orchestration