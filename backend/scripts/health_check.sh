#!/bin/bash

# Health Check Script for AI Agent Platform
# This script performs comprehensive health checks on the deployed application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"
TIMEOUT=30
RETRY_COUNT=3

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

check_service_health() {
    local service_name=$1
    local url=$2
    local endpoint=${3:-"/health"}
    
    log_info "Checking $service_name health..."
    
    for i in $(seq 1 $RETRY_COUNT); do
        if curl -f -s --max-time $TIMEOUT "$url$endpoint" > /dev/null 2>&1; then
            log_success "$service_name is healthy"
            return 0
        else
            log_warning "$service_name health check failed (attempt $i/$RETRY_COUNT)"
            if [ $i -lt $RETRY_COUNT ]; then
                sleep 5
            fi
        fi
    done
    
    log_error "$service_name is unhealthy"
    return 1
}

check_database_connection() {
    log_info "Checking database connection..."
    
    # Check if PostgreSQL container is running
    if docker ps | grep -q postgres; then
        log_success "PostgreSQL container is running"
    else
        log_error "PostgreSQL container is not running"
        return 1
    fi
    
    # Check database connectivity from backend
    response=$(curl -s --max-time $TIMEOUT "$BACKEND_URL/health" 2>/dev/null || echo "failed")
    if [[ "$response" == *"healthy"* ]] || [[ "$response" == *"status"* ]]; then
        log_success "Database connection is healthy"
        return 0
    else
        log_error "Database connection check failed"
        return 1
    fi
}

check_api_endpoints() {
    log_info "Checking API endpoints..."
    
    local endpoints=(
        "/"
        "/health"
        "/api/v1/auth/me"
        "/docs"
    )
    
    local failed_endpoints=()
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s --max-time $TIMEOUT "$BACKEND_URL$endpoint" > /dev/null 2>&1; then
            log_success "Endpoint $endpoint is accessible"
        else
            log_warning "Endpoint $endpoint is not accessible"
            failed_endpoints+=("$endpoint")
        fi
    done
    
    if [ ${#failed_endpoints[@]} -eq 0 ]; then
        log_success "All API endpoints are accessible"
        return 0
    else
        log_error "Some API endpoints are not accessible: ${failed_endpoints[*]}"
        return 1
    fi
}

check_docker_containers() {
    log_info "Checking Docker containers..."
    
    local containers=(
        "ai-agent-platform-backend-1"
        "ai-agent-platform-postgres-1"
        "ai-agent-platform-redis-1"
    )
    
    local failed_containers=()
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "$container"; then
            log_success "Container $container is running"
        else
            log_warning "Container $container is not running"
            failed_containers+=("$container")
        fi
    done
    
    if [ ${#failed_containers[@]} -eq 0 ]; then
        log_success "All required containers are running"
        return 0
    else
        log_error "Some containers are not running: ${failed_containers[*]}"
        return 1
    fi
}

check_disk_space() {
    log_info "Checking disk space..."
    
    local usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ $usage -lt 80 ]; then
        log_success "Disk space is sufficient ($usage% used)"
        return 0
    elif [ $usage -lt 90 ]; then
        log_warning "Disk space is getting low ($usage% used)"
        return 0
    else
        log_error "Disk space is critically low ($usage% used)"
        return 1
    fi
}

check_memory_usage() {
    log_info "Checking memory usage..."
    
    local usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    
    if [ $usage -lt 80 ]; then
        log_success "Memory usage is normal ($usage%)"
        return 0
    elif [ $usage -lt 90 ]; then
        log_warning "Memory usage is high ($usage%)"
        return 0
    else
        log_error "Memory usage is critically high ($usage%)"
        return 1
    fi
}

check_log_errors() {
    log_info "Checking for recent errors in logs..."
    
    local error_count=0
    
    # Check backend logs for errors
    if [ -d "logs/backend" ]; then
        local backend_errors=$(find logs/backend -name "*.log" -mtime -1 -exec grep -l "ERROR\|CRITICAL" {} \; 2>/dev/null | wc -l)
        if [ $backend_errors -gt 0 ]; then
            log_warning "Found $backend_errors backend log files with errors"
            error_count=$((error_count + backend_errors))
        fi
    fi
    
    # Check Docker logs for errors
    local docker_errors=$(docker-compose logs --tail=100 2>/dev/null | grep -c "ERROR\|CRITICAL" || echo "0")
    if [ $docker_errors -gt 0 ]; then
        log_warning "Found $docker_errors errors in Docker logs"
        error_count=$((error_count + docker_errors))
    fi
    
    if [ $error_count -eq 0 ]; then
        log_success "No recent errors found in logs"
        return 0
    else
        log_warning "Found $error_count recent errors in logs"
        return 0  # Don't fail health check for log errors
    fi
}

generate_health_report() {
    local report_file="health_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "Generating health report: $report_file"
    
    {
        echo "AI Agent Platform Health Report"
        echo "Generated: $(date)"
        echo "=================================="
        echo ""
        
        echo "System Information:"
        echo "  Hostname: $(hostname)"
        echo "  Uptime: $(uptime)"
        echo "  Load Average: $(cat /proc/loadavg 2>/dev/null || echo 'N/A')"
        echo ""
        
        echo "Docker Status:"
        docker-compose ps
        echo ""
        
        echo "Container Resource Usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
        echo ""
        
        echo "Disk Usage:"
        df -h
        echo ""
        
        echo "Memory Usage:"
        free -h
        echo ""
        
        echo "Recent Logs (last 50 lines):"
        docker-compose logs --tail=50
        echo ""
        
    } > "$report_file"
    
    log_success "Health report generated: $report_file"
}

# Main health check process
main() {
    log_info "Starting comprehensive health check..."
    
    local overall_status=0
    
    # Run all health checks
    check_docker_containers || overall_status=1
    check_service_health "Backend" "$BACKEND_URL" "/health" || overall_status=1
    check_service_health "Frontend" "$FRONTEND_URL" "" || overall_status=1
    check_database_connection || overall_status=1
    check_api_endpoints || overall_status=1
    check_disk_space || overall_status=1
    check_memory_usage || overall_status=1
    check_log_errors  # Don't fail for log errors
    
    # Generate report
    generate_health_report
    
    # Final status
    if [ $overall_status -eq 0 ]; then
        log_success "All health checks passed! System is healthy."
        exit 0
    else
        log_error "Some health checks failed! System needs attention."
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  help, -h, --help    Show this help message"
        echo "  report              Generate health report only"
        echo "  quick               Run quick health check (skip detailed checks)"
        echo ""
        echo "Examples:"
        echo "  $0                  # Full health check"
        echo "  $0 report           # Generate report only"
        echo "  $0 quick            # Quick health check"
        exit 0
        ;;
    "report")
        generate_health_report
        exit 0
        ;;
    "quick")
        log_info "Running quick health check..."
        check_docker_containers
        check_service_health "Backend" "$BACKEND_URL" "/health"
        check_service_health "Frontend" "$FRONTEND_URL" ""
        log_success "Quick health check completed"
        exit 0
        ;;
    *)
        main
        ;;
esac
