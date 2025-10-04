#!/bin/bash

# Restore Script for AI Agent Platform
# This script restores the application from a backup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="/var/backups/ai-agent-platform"

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

list_available_backups() {
    log_info "Available backups:"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_DIR"
        return 1
    fi
    
    local backup_count=0
    for backup_file in "$BACKUP_DIR"/backup_*.tar.gz; do
        if [ -f "$backup_file" ]; then
            local backup_name=$(basename "$backup_file" .tar.gz)
            local backup_size=$(du -h "$backup_file" | cut -f1)
            local backup_date=$(stat -c %y "$backup_file" | cut -d' ' -f1)
            echo "  $backup_name ($backup_size, $backup_date)"
            backup_count=$((backup_count + 1))
        fi
    done
    
    if [ $backup_count -eq 0 ]; then
        log_warning "No backups found"
        return 1
    fi
    
    return 0
}

verify_backup() {
    local backup_name=$1
    local backup_file="$BACKUP_DIR/${backup_name}.tar.gz"
    
    log_info "Verifying backup: $backup_name"
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Test archive integrity
    if tar -tzf "$backup_file" > /dev/null 2>&1; then
        log_success "Backup archive integrity verified"
        return 0
    else
        log_error "Backup archive is corrupted"
        return 1
    fi
}

extract_backup() {
    local backup_name=$1
    local backup_file="$BACKUP_DIR/${backup_name}.tar.gz"
    local extract_dir="/tmp/restore_$backup_name"
    
    log_info "Extracting backup: $backup_name"
    
    # Create extraction directory
    mkdir -p "$extract_dir"
    
    # Extract backup
    tar -xzf "$backup_file" -C "$extract_dir"
    
    log_success "Backup extracted to: $extract_dir"
    echo "$extract_dir"
}

stop_services() {
    log_info "Stopping application services..."
    
    # Stop Docker containers
    docker-compose down || true
    
    log_success "Services stopped"
}

restore_database() {
    local extract_dir=$1
    
    log_info "Restoring database..."
    
    local db_dump_file="$extract_dir/database.sql.gz"
    
    if [ ! -f "$db_dump_file" ]; then
        log_error "Database dump not found: $db_dump_file"
        return 1
    fi
    
    # Start PostgreSQL container if not running
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Get database credentials
    local db_host=${DB_HOST:-localhost}
    local db_port=${DB_PORT:-5432}
    local db_name=${DB_NAME:-ai_agent_platform}
    local db_user=${DB_USER:-postgres}
    
    # Drop existing database and recreate
    log_info "Recreating database..."
    docker exec ai-agent-platform-postgres-1 psql -h localhost -U "$db_user" -c "DROP DATABASE IF EXISTS $db_name;" postgres
    docker exec ai-agent-platform-postgres-1 psql -h localhost -U "$db_user" -c "CREATE DATABASE $db_name;" postgres
    
    # Restore database
    log_info "Restoring database from dump..."
    gunzip -c "$db_dump_file" | docker exec -i ai-agent-platform-postgres-1 psql -h localhost -U "$db_user" "$db_name"
    
    log_success "Database restored successfully"
}

restore_uploads() {
    local extract_dir=$1
    
    log_info "Restoring uploads..."
    
    local backup_uploads_dir="$extract_dir/uploads"
    local target_uploads_dir="backend/uploads"
    
    if [ -d "$backup_uploads_dir" ]; then
        # Create target directory if it doesn't exist
        mkdir -p "$target_uploads_dir"
        
        # Copy uploads
        cp -r "$backup_uploads_dir"/* "$target_uploads_dir/" 2>/dev/null || true
        
        log_success "Uploads restored successfully"
    else
        log_warning "Uploads backup not found"
    fi
}

restore_configurations() {
    local extract_dir=$1
    
    log_info "Restoring configurations..."
    
    local backup_config_dir="$extract_dir/config"
    
    if [ -d "$backup_config_dir" ]; then
        # Restore configuration files
        for config_file in "$backup_config_dir"/*; do
            if [ -f "$config_file" ]; then
                local filename=$(basename "$config_file")
                local target_file=""
                
                case "$filename" in
                    ".env")
                        target_file="backend/.env"
                        ;;
                    "alembic.ini")
                        target_file="backend/alembic.ini"
                        ;;
                    "docker-compose.yml")
                        target_file="docker-compose.yml"
                        ;;
                    "requirements.txt")
                        target_file="backend/requirements.txt"
                        ;;
                    "package.json")
                        target_file="frontend/package.json"
                        ;;
                esac
                
                if [ -n "$target_file" ]; then
                    cp "$config_file" "$target_file"
                    log_success "Restored: $target_file"
                fi
            fi
        done
        
        log_success "Configurations restored successfully"
    else
        log_warning "Configuration backup not found"
    fi
}

run_database_migrations() {
    log_info "Running database migrations..."
    
    cd backend
    
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

start_services() {
    log_info "Starting application services..."
    
    # Start all services
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
        return 1
    fi
    
    log_success "Services started successfully"
}

cleanup_extraction() {
    local extract_dir=$1
    
    log_info "Cleaning up extraction directory..."
    
    rm -rf "$extract_dir"
    
    log_success "Cleanup completed"
}

show_restore_info() {
    local backup_name=$1
    
    log_success "Restore completed successfully!"
    echo ""
    echo "Restore Information:"
    echo "  Backup: $backup_name"
    echo "  Date: $(date)"
    echo ""
    echo "Application URLs:"
    echo "  Backend API: http://localhost:8000"
    echo "  Frontend: http://localhost:3000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo ""
    echo "Useful commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Health check: ./scripts/health_check.sh"
    echo ""
}

# Main restore process
main() {
    local backup_name=${1:-}
    
    if [ -z "$backup_name" ]; then
        log_error "Backup name is required"
        echo ""
        echo "Usage: $0 <backup_name>"
        echo ""
        echo "Available backups:"
        list_available_backups
        exit 1
    fi
    
    log_info "Starting restore process for backup: $backup_name"
    
    # Verify backup exists and is valid
    verify_backup "$backup_name" || exit 1
    
    # Extract backup
    local extract_dir=$(extract_backup "$backup_name")
    
    # Stop services
    stop_services
    
    # Restore components
    restore_database "$extract_dir"
    restore_uploads "$extract_dir"
    restore_configurations "$extract_dir"
    
    # Run migrations
    run_database_migrations
    
    # Start services
    start_services
    
    # Cleanup
    cleanup_extraction "$extract_dir"
    
    # Show restore info
    show_restore_info "$backup_name"
    
    log_success "Restore process completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "Usage: $0 [backup_name]"
        echo ""
        echo "Arguments:"
        echo "  backup_name    Name of the backup to restore from"
        echo ""
        echo "Options:"
        echo "  help, -h, --help    Show this help message"
        echo "  list                List available backups"
        echo "  verify <name>       Verify backup integrity"
        echo ""
        echo "Examples:"
        echo "  $0 backup_20240103_120000  # Restore from specific backup"
        echo "  $0 list                     # List available backups"
        echo "  $0 verify backup_20240103_120000  # Verify backup"
        exit 0
        ;;
    "list")
        list_available_backups
        exit $?
        ;;
    "verify")
        if [ -z "${2:-}" ]; then
            log_error "Backup name required for verification"
            exit 1
        fi
        verify_backup "$2"
        exit $?
        ;;
    *)
        main "$1"
        ;;
esac
