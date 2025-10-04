#!/bin/bash

# Backup Script for AI Agent Platform
# This script creates comprehensive backups of the application data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="/var/backups/ai-agent-platform"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_$DATE"
RETENTION_DAYS=30

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

create_backup_directory() {
    log_info "Creating backup directory..."
    
    if [ ! -d "$BACKUP_DIR" ]; then
        sudo mkdir -p "$BACKUP_DIR"
        sudo chown $(whoami):$(whoami) "$BACKUP_DIR"
    fi
    
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
    log_success "Backup directory created: $BACKUP_DIR/$BACKUP_NAME"
}

backup_database() {
    log_info "Backing up database..."
    
    # Get database credentials from environment
    local db_host=${DB_HOST:-localhost}
    local db_port=${DB_PORT:-5432}
    local db_name=${DB_NAME:-ai_agent_platform}
    local db_user=${DB_USER:-postgres}
    
    # Create database dump
    local db_backup_file="$BACKUP_DIR/$BACKUP_NAME/database.sql"
    
    if docker exec ai-agent-platform-postgres-1 pg_dump -h localhost -U "$db_user" "$db_name" > "$db_backup_file" 2>/dev/null; then
        log_success "Database backup completed: $db_backup_file"
        
        # Compress database backup
        gzip "$db_backup_file"
        log_success "Database backup compressed"
    else
        log_error "Database backup failed"
        return 1
    fi
}

backup_uploads() {
    log_info "Backing up uploads..."
    
    local uploads_dir="backend/uploads"
    local backup_uploads_dir="$BACKUP_DIR/$BACKUP_NAME/uploads"
    
    if [ -d "$uploads_dir" ]; then
        cp -r "$uploads_dir" "$backup_uploads_dir"
        log_success "Uploads backup completed: $backup_uploads_dir"
    else
        log_warning "Uploads directory not found: $uploads_dir"
    fi
}

backup_configurations() {
    log_info "Backing up configurations..."
    
    local config_files=(
        "backend/.env"
        "backend/alembic.ini"
        "docker-compose.yml"
        "backend/requirements.txt"
        "frontend/package.json"
    )
    
    local config_backup_dir="$BACKUP_DIR/$BACKUP_NAME/config"
    mkdir -p "$config_backup_dir"
    
    for config_file in "${config_files[@]}"; do
        if [ -f "$config_file" ]; then
            cp "$config_file" "$config_backup_dir/"
            log_success "Backed up: $config_file"
        else
            log_warning "Configuration file not found: $config_file"
        fi
    done
}

backup_logs() {
    log_info "Backing up logs..."
    
    local logs_backup_dir="$BACKUP_DIR/$BACKUP_NAME/logs"
    mkdir -p "$logs_backup_dir"
    
    # Backup application logs
    if [ -d "logs" ]; then
        cp -r logs/* "$logs_backup_dir/" 2>/dev/null || true
        log_success "Application logs backed up"
    fi
    
    # Backup Docker logs
    local docker_logs_file="$logs_backup_dir/docker_logs.txt"
    docker-compose logs > "$docker_logs_file" 2>/dev/null || true
    log_success "Docker logs backed up"
}

backup_code() {
    log_info "Backing up application code..."
    
    local code_backup_dir="$BACKUP_DIR/$BACKUP_NAME/code"
    mkdir -p "$code_backup_dir"
    
    # Create a tar archive of the application code
    tar -czf "$code_backup_dir/application_code.tar.gz" \
        --exclude="node_modules" \
        --exclude="__pycache__" \
        --exclude=".git" \
        --exclude="*.pyc" \
        --exclude=".env" \
        --exclude="logs" \
        --exclude="uploads" \
        . 2>/dev/null || true
    
    log_success "Application code backed up: $code_backup_dir/application_code.tar.gz"
}

create_backup_manifest() {
    log_info "Creating backup manifest..."
    
    local manifest_file="$BACKUP_DIR/$BACKUP_NAME/manifest.txt"
    
    {
        echo "AI Agent Platform Backup Manifest"
        echo "=================================="
        echo "Backup Date: $(date)"
        echo "Backup Name: $BACKUP_NAME"
        echo "Backup Directory: $BACKUP_DIR/$BACKUP_NAME"
        echo ""
        
        echo "Included Components:"
        echo "  - Database dump (compressed)"
        echo "  - Upload files"
        echo "  - Configuration files"
        echo "  - Application logs"
        echo "  - Docker logs"
        echo "  - Application code"
        echo ""
        
        echo "System Information:"
        echo "  Hostname: $(hostname)"
        echo "  OS: $(uname -a)"
        echo "  Docker Version: $(docker --version)"
        echo "  Disk Usage: $(df -h / | awk 'NR==2 {print $5}')"
        echo ""
        
        echo "Backup Contents:"
        find "$BACKUP_DIR/$BACKUP_NAME" -type f -exec ls -lh {} \; | while read line; do
            echo "  $line"
        done
        
    } > "$manifest_file"
    
    log_success "Backup manifest created: $manifest_file"
}

compress_backup() {
    log_info "Compressing backup..."
    
    local backup_archive="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    
    cd "$BACKUP_DIR"
    tar -czf "$backup_archive" "$BACKUP_NAME"
    cd - > /dev/null
    
    # Remove uncompressed backup directory
    rm -rf "$BACKUP_DIR/$BACKUP_NAME"
    
    log_success "Backup compressed: $backup_archive"
    
    # Show backup size
    local backup_size=$(du -h "$backup_archive" | cut -f1)
    log_info "Backup size: $backup_size"
}

cleanup_old_backups() {
    log_info "Cleaning up old backups (older than $RETENTION_DAYS days)..."
    
    local deleted_count=0
    
    # Find and delete old backup files
    while IFS= read -r -d '' backup_file; do
        rm "$backup_file"
        deleted_count=$((deleted_count + 1))
        log_info "Deleted old backup: $(basename "$backup_file")"
    done < <(find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +$RETENTION_DAYS -print0)
    
    if [ $deleted_count -eq 0 ]; then
        log_info "No old backups to clean up"
    else
        log_success "Cleaned up $deleted_count old backups"
    fi
}

verify_backup() {
    log_info "Verifying backup integrity..."
    
    local backup_archive="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    
    if [ -f "$backup_archive" ]; then
        # Test archive integrity
        if tar -tzf "$backup_archive" > /dev/null 2>&1; then
            log_success "Backup archive integrity verified"
            return 0
        else
            log_error "Backup archive is corrupted"
            return 1
        fi
    else
        log_error "Backup archive not found: $backup_archive"
        return 1
    fi
}

show_backup_info() {
    log_success "Backup completed successfully!"
    echo ""
    echo "Backup Information:"
    echo "  Name: $BACKUP_NAME"
    echo "  Location: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    echo "  Size: $(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)"
    echo "  Date: $(date)"
    echo ""
    echo "To restore from this backup:"
    echo "  ./scripts/restore.sh $BACKUP_NAME"
    echo ""
}

# Main backup process
main() {
    log_info "Starting backup process..."
    
    create_backup_directory
    backup_database
    backup_uploads
    backup_configurations
    backup_logs
    backup_code
    create_backup_manifest
    compress_backup
    cleanup_old_backups
    verify_backup
    show_backup_info
    
    log_success "Backup process completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  help, -h, --help    Show this help message"
        echo "  list                List available backups"
        echo "  verify <name>       Verify backup integrity"
        echo "  cleanup             Clean up old backups only"
        echo ""
        echo "Examples:"
        echo "  $0                  # Create new backup"
        echo "  $0 list             # List available backups"
        echo "  $0 verify backup_20240103_120000  # Verify specific backup"
        echo "  $0 cleanup          # Clean up old backups"
        exit 0
        ;;
    "list")
        log_info "Available backups:"
        if [ -d "$BACKUP_DIR" ]; then
            ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null | while read line; do
                echo "  $line"
            done
        else
            log_warning "No backups found"
        fi
        exit 0
        ;;
    "verify")
        if [ -z "${2:-}" ]; then
            log_error "Backup name required for verification"
            exit 1
        fi
        BACKUP_NAME="$2"
        verify_backup
        exit $?
        ;;
    "cleanup")
        cleanup_old_backups
        exit 0
        ;;
    *)
        main
        ;;
esac
