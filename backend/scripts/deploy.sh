#!/bin/bash

# Production Deployment Script for AI Agent Platform
# This script handles the complete deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-agent-platform"
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
ENVIRONMENT=${1:-production}
SKIP_TESTS=${2:-false}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml not found. Please run from project root."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker."
        exit 1
    fi
    
    # Check if required tools are installed
    command -v python3 >/dev/null 2>&1 || { log_error "Python3 is required but not installed."; exit 1; }
    command -v npm >/dev/null 2>&1 || { log_error "npm is required but not installed."; exit 1; }
    
    log_success "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment for $ENVIRONMENT..."
    
    # Copy environment files if they don't exist
    if [ ! -f "$BACKEND_DIR/.env" ]; then
        if [ -f "$BACKEND_DIR/.env.example" ]; then
            cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
            log_warning "Created .env from .env.example. Please update with your values."
        else
            log_error ".env.example not found. Please create .env file manually."
            exit 1
        fi
    fi
    
    # Set environment-specific configurations
    case $ENVIRONMENT in
        "production")
            export ENVIRONMENT=production
            export DEBUG=false
            ;;
        "staging")
            export ENVIRONMENT=staging
            export DEBUG=false
            ;;
        "development")
            export ENVIRONMENT=development
            export DEBUG=true
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Use: production, staging, or development"
            exit 1
            ;;
    esac
    
    log_success "Environment setup completed"
}

install_dependencies() {
    log_info "Installing dependencies..."
    
    # Install backend dependencies
    log_info "Installing Python dependencies..."
    cd "$BACKEND_DIR"
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
    else
        log_warning "requirements.txt not found, skipping Python dependencies"
    fi
    
    # Install frontend dependencies
    cd "../$FRONTEND_DIR"
    log_info "Installing Node.js dependencies..."
    if [ -f "package.json" ]; then
        npm install
    else
        log_warning "package.json not found, skipping Node.js dependencies"
    fi
    
    cd ..
    log_success "Dependencies installation completed"
}

run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        log_warning "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running tests..."
    
    # Run backend tests
    cd "$BACKEND_DIR"
    if [ -d "tests" ]; then
        log_info "Running backend tests..."
        python3 -m pytest tests/test_production_improvements.py -v --tb=short
        if [ $? -ne 0 ]; then
            log_error "Backend tests failed"
            exit 1
        fi
    else
        log_warning "No tests directory found in backend"
    fi
    
    # Run frontend tests
    cd "../$FRONTEND_DIR"
    if [ -f "package.json" ] && grep -q '"test"' package.json; then
        log_info "Running frontend tests..."
        npm test -- --watchAll=false
        if [ $? -ne 0 ]; then
            log_error "Frontend tests failed"
            exit 1
        fi
    else
        log_warning "No test script found in frontend"
    fi
    
    cd ..
    log_success "All tests passed"
}

run_database_migrations() {
    log_info "Running database migrations..."
    
    cd "$BACKEND_DIR"
    
    # Check if alembic is available
    if command -v alembic >/dev/null 2>&1; then
        alembic upgrade head
    elif python3 -m alembic >/dev/null 2>&1; then
        python3 -m alembic upgrade head
    else
        log_warning "Alembic not found, skipping migrations"
    fi
    
    cd ..
    log_success "Database migrations completed"
}

build_application() {
    log_info "Building application..."
    
    # Build frontend
    cd "$FRONTEND_DIR"
    if [ -f "package.json" ]; then
        log_info "Building frontend..."
        npm run build
        if [ $? -ne 0 ]; then
            log_error "Frontend build failed"
            exit 1
        fi
    else
        log_warning "No package.json found in frontend"
    fi
    
    cd ..
    
    # Build backend Docker image
    log_info "Building backend Docker image..."
    docker build -t "$PROJECT_NAME-backend:$ENVIRONMENT" "$BACKEND_DIR"
    if [ $? -ne 0 ]; then
        log_error "Backend Docker build failed"
        exit 1
    fi
    
    log_success "Application build completed"
}

deploy_application() {
    log_info "Deploying application..."
    
    # Stop existing containers
    log_info "Stopping existing containers..."
    docker-compose down || true
    
    # Start services
    log_info "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    log_info "Performing health check..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Backend health check passed"
    else
        log_error "Backend health check failed"
        exit 1
    fi
    
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        log_success "Frontend health check passed"
    else
        log_warning "Frontend health check failed (may not be configured)"
    fi
    
    log_success "Application deployment completed"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create log directories
    mkdir -p logs/backend
    mkdir -p logs/frontend
    
    # Set up log rotation
    if [ -f "/etc/logrotate.d/$PROJECT_NAME" ]; then
        log_info "Log rotation already configured"
    else
        log_info "Setting up log rotation..."
        sudo tee /etc/logrotate.d/$PROJECT_NAME > /dev/null <<EOF
logs/backend/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
}

logs/frontend/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF
    fi
    
    log_success "Monitoring setup completed"
}

cleanup() {
    log_info "Cleaning up..."
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove unused Docker volumes
    docker volume prune -f
    
    log_success "Cleanup completed"
}

show_deployment_info() {
    log_success "Deployment completed successfully!"
    echo ""
    echo "Application URLs:"
    echo "  Backend API: http://localhost:8000"
    echo "  Frontend: http://localhost:3000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo ""
    echo "Useful commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Restart services: docker-compose restart"
    echo "  Update application: ./scripts/deploy.sh $ENVIRONMENT"
    echo ""
}

# Main deployment process
main() {
    log_info "Starting deployment process for $ENVIRONMENT environment..."
    
    check_prerequisites
    setup_environment
    install_dependencies
    run_tests
    run_database_migrations
    build_application
    deploy_application
    setup_monitoring
    cleanup
    show_deployment_info
    
    log_success "Deployment process completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "Usage: $0 [environment] [skip-tests]"
        echo ""
        echo "Arguments:"
        echo "  environment   Target environment (production, staging, development)"
        echo "  skip-tests    Set to 'true' to skip running tests"
        echo ""
        echo "Examples:"
        echo "  $0 production"
        echo "  $0 staging true"
        echo "  $0 development"
        exit 0
        ;;
    *)
        main
        ;;
esac
