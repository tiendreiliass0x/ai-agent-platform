#!/bin/bash

# Monitoring Script for AI Agent Platform
# This script provides real-time monitoring and alerting

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
LOG_FILE="/var/log/ai-agent-platform/monitor.log"
ALERT_EMAIL=${ALERT_EMAIL:-""}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
MAX_RETRIES=3

# Functions
log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

setup_logging() {
    # Create log directory
    sudo mkdir -p "$(dirname "$LOG_FILE")"
    sudo touch "$LOG_FILE"
    sudo chown $(whoami):$(whoami) "$LOG_FILE"
}

send_alert() {
    local subject=$1
    local message=$2
    
    if [ -n "$ALERT_EMAIL" ]; then
        echo "$message" | mail -s "$subject" "$ALERT_EMAIL" 2>/dev/null || true
    fi
    
    # Log alert
    log_error "ALERT: $subject - $message"
}

check_service_health() {
    local service_name=$1
    local url=$2
    local endpoint=${3:-"/health"}
    
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if curl -f -s --max-time 10 "$url$endpoint" > /dev/null 2>&1; then
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        sleep 5
    done
    
    send_alert "Service Down" "$service_name is not responding at $url$endpoint"
    return 1
}

check_database_health() {
    # Check if PostgreSQL container is running
    if ! docker ps | grep -q postgres; then
        send_alert "Database Down" "PostgreSQL container is not running"
        return 1
    fi
    
    # Check database connectivity
    if ! check_service_health "Database" "$BACKEND_URL" "/health"; then
        send_alert "Database Connection" "Database connectivity check failed"
        return 1
    fi
    
    return 0
}

check_disk_space() {
    local usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ $usage -gt 90 ]; then
        send_alert "Disk Space Critical" "Disk usage is at ${usage}%"
        return 1
    elif [ $usage -gt 80 ]; then
        log_warning "Disk usage is high: ${usage}%"
    fi
    
    return 0
}

check_memory_usage() {
    local usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    
    if [ $usage -gt 90 ]; then
        send_alert "Memory Critical" "Memory usage is at ${usage}%"
        return 1
    elif [ $usage -gt 80 ]; then
        log_warning "Memory usage is high: ${usage}%"
    fi
    
    return 0
}

check_cpu_usage() {
    local usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    
    if [ $usage -gt 90 ]; then
        send_alert "CPU Critical" "CPU usage is at ${usage}%"
        return 1
    elif [ $usage -gt 80 ]; then
        log_warning "CPU usage is high: ${usage}%"
    fi
    
    return 0
}

check_docker_containers() {
    local containers=(
        "ai-agent-platform-backend-1"
        "ai-agent-platform-postgres-1"
        "ai-agent-platform-redis-1"
    )
    
    local failed_containers=()
    
    for container in "${containers[@]}"; do
        if ! docker ps --format "table {{.Names}}" | grep -q "$container"; then
            failed_containers+=("$container")
        fi
    done
    
    if [ ${#failed_containers[@]} -gt 0 ]; then
        send_alert "Container Down" "Containers not running: ${failed_containers[*]}"
        return 1
    fi
    
    return 0
}

check_log_errors() {
    local error_count=0
    
    # Check for recent errors in Docker logs
    local recent_errors=$(docker-compose logs --tail=100 2>/dev/null | grep -c "ERROR\|CRITICAL" || echo "0")
    
    if [ $recent_errors -gt 10 ]; then
        send_alert "High Error Rate" "Found $recent_errors recent errors in logs"
        error_count=$((error_count + recent_errors))
    fi
    
    # Check application logs
    if [ -d "logs" ]; then
        local app_errors=$(find logs -name "*.log" -mtime -1 -exec grep -l "ERROR\|CRITICAL" {} \; 2>/dev/null | wc -l)
        if [ $app_errors -gt 5 ]; then
            log_warning "Found $app_errors application log files with errors"
            error_count=$((error_count + app_errors))
        fi
    fi
    
    return 0  # Don't fail monitoring for log errors
}

check_api_response_time() {
    local start_time=$(date +%s%N)
    
    if curl -f -s --max-time 30 "$BACKEND_URL/health" > /dev/null 2>&1; then
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 ))  # Convert to milliseconds
        
        if [ $response_time -gt 5000 ]; then
            send_alert "Slow Response" "API response time is ${response_time}ms"
            return 1
        elif [ $response_time -gt 2000 ]; then
            log_warning "API response time is slow: ${response_time}ms"
        fi
    else
        send_alert "API Unavailable" "API health check failed"
        return 1
    fi
    
    return 0
}

generate_status_report() {
    local report_file="/tmp/status_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "AI Agent Platform Status Report"
        echo "Generated: $(date)"
        echo "=================================="
        echo ""
        
        echo "System Information:"
        echo "  Hostname: $(hostname)"
        echo "  Uptime: $(uptime)"
        echo "  Load Average: $(cat /proc/loadavg 2>/dev/null || echo 'N/A')"
        echo ""
        
        echo "Resource Usage:"
        echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')"
        echo "  Memory: $(free | awk 'NR==2{printf "%.1f%%", $3*100/$2}')"
        echo "  Disk: $(df / | awk 'NR==2 {print $5}')"
        echo ""
        
        echo "Docker Status:"
        docker-compose ps
        echo ""
        
        echo "Container Resource Usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
        echo ""
        
        echo "Recent Logs (last 20 lines):"
        docker-compose logs --tail=20
        echo ""
        
    } > "$report_file"
    
    log_info "Status report generated: $report_file"
}

run_health_check() {
    local overall_status=0
    
    log_info "Running comprehensive health check..."
    
    # Service health checks
    check_service_health "Backend" "$BACKEND_URL" "/health" || overall_status=1
    check_service_health "Frontend" "$FRONTEND_URL" "" || overall_status=1
    check_database_health || overall_status=1
    
    # System resource checks
    check_disk_space || overall_status=1
    check_memory_usage || overall_status=1
    check_cpu_usage || overall_status=1
    
    # Container checks
    check_docker_containers || overall_status=1
    
    # Performance checks
    check_api_response_time || overall_status=1
    
    # Log checks
    check_log_errors
    
    if [ $overall_status -eq 0 ]; then
        log_success "All health checks passed"
    else
        log_error "Some health checks failed"
        generate_status_report
    fi
    
    return $overall_status
}

monitor_loop() {
    log_info "Starting monitoring loop (interval: ${CHECK_INTERVAL}s)"
    
    while true; do
        run_health_check
        sleep $CHECK_INTERVAL
    done
}

show_help() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start              Start continuous monitoring"
    echo "  check              Run single health check"
    echo "  report             Generate status report"
    echo "  help, -h, --help   Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ALERT_EMAIL        Email address for alerts"
    echo "  CHECK_INTERVAL     Check interval in seconds (default: 60)"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start continuous monitoring"
    echo "  $0 check                    # Run single health check"
    echo "  $0 report                   # Generate status report"
    echo "  ALERT_EMAIL=admin@example.com $0 start  # Start with email alerts"
    echo ""
}

# Main script logic
main() {
    # Setup logging
    setup_logging
    
    case "${1:-start}" in
        "start")
            monitor_loop
            ;;
        "check")
            run_health_check
            ;;
        "report")
            generate_status_report
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Handle script arguments
main "$@"
