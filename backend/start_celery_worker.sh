#!/bin/bash
# Celery Worker Startup Script
# This script starts multiple Celery workers for different priority queues

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Celery Workers for AI Agent Platform${NC}"
echo -e "${BLUE}========================================${NC}"

# macOS + PyTorch (MPS) can crash under forked pools. Disable fork safety or
# switch to solo pool when running locally on Darwin for stability.
if [[ "$OSTYPE" == "darwin"* ]]; then
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    CELERY_WORKER_POOL="${CELERY_WORKER_POOL:-solo}"
else
    CELERY_WORKER_POOL="${CELERY_WORKER_POOL:-prefork}"
fi

# Configuration
APP_NAME="app.celery_app"
LOG_LEVEL="${CELERY_LOG_LEVEL:-info}"
WORKER_CONCURRENCY_HIGH="${WORKER_CONCURRENCY_HIGH:-4}"
WORKER_CONCURRENCY_MEDIUM="${WORKER_CONCURRENCY_MEDIUM:-8}"
WORKER_CONCURRENCY_LOW="${WORKER_CONCURRENCY_LOW:-16}"

# Check if running in Docker or local
if [ -f "/.dockerenv" ]; then
    echo -e "${YELLOW}Running in Docker environment${NC}"
    PYTHON_CMD="python"
else
    echo -e "${YELLOW}Running in local environment${NC}"
    # Try to activate virtual environment if it exists
    if [ -f "venv-clean/bin/activate" ]; then
        source venv-clean/bin/activate
        echo -e "${GREEN}Virtual environment activated${NC}"
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo -e "${GREEN}Virtual environment activated${NC}"
    fi
    PYTHON_CMD="python3"
fi

# Verify Celery is installed
if ! $PYTHON_CMD -c "import celery" 2>/dev/null; then
    echo -e "${YELLOW}ERROR: Celery not installed. Installing...${NC}"
    pip install celery redis
fi

# Verify Redis connection
echo -e "${BLUE}Checking Redis connection...${NC}"
if ! $PYTHON_CMD -c "import redis; r = redis.from_url('${REDIS_URL:-redis://localhost:6379}'); r.ping()" 2>/dev/null; then
    echo -e "${YELLOW}WARNING: Cannot connect to Redis. Make sure Redis is running.${NC}"
    echo -e "${YELLOW}Expected Redis URL: ${REDIS_URL:-redis://localhost:6379}${NC}"
fi

# Function to start a worker
start_worker() {
    local queue=$1
    local concurrency=$2
    local worker_name=$3

    echo -e "${GREEN}Starting ${worker_name} worker for queue '${queue}' with concurrency ${concurrency}${NC}"

    celery -A $APP_NAME worker \
        --queues=$queue \
        --concurrency=$concurrency \
        --pool=$CELERY_WORKER_POOL \
        --loglevel=$LOG_LEVEL \
        --hostname=${worker_name}@%h \
        --max-tasks-per-child=1000 \
        --time-limit=3600 \
        --soft-time-limit=3000 \
        --pidfile=/tmp/celery-${worker_name}.pid \
        &
}

# Start workers for different priority queues
echo -e "${BLUE}Launching workers...${NC}"

# High priority worker - document processing (less concurrency, more resources per task)
start_worker "high_priority" $WORKER_CONCURRENCY_HIGH "high_priority"

# Medium priority worker - embeddings, analytics (balanced)
start_worker "medium_priority" $WORKER_CONCURRENCY_MEDIUM "medium_priority"

# Low priority worker - crawling, batch operations (high concurrency)
start_worker "low_priority" $WORKER_CONCURRENCY_LOW "low_priority"

# Default queue worker
start_worker "default" 4 "default"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All Celery workers started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Worker Configuration:${NC}"
echo -e "  - High Priority Queue: ${WORKER_CONCURRENCY_HIGH} workers (document processing)"
echo -e "  - Medium Priority Queue: ${WORKER_CONCURRENCY_MEDIUM} workers (embeddings, analytics)"
echo -e "  - Low Priority Queue: ${WORKER_CONCURRENCY_LOW} workers (crawling, batch ops)"
echo -e "  - Default Queue: 4 workers"
echo ""
echo -e "${BLUE}Monitoring:${NC}"
echo -e "  - View workers: celery -A app.celery_app inspect active"
echo -e "  - View stats: celery -A app.celery_app inspect stats"
echo -e "  - Kill workers: pkill -f 'celery worker'"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all workers${NC}"

# Wait for all background jobs
wait
