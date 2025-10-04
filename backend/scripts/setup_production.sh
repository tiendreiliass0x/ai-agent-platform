#!/bin/bash

# Production Setup Script for AI Agent Platform
# This script sets up the production environment

set -e

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
ENVIRONMENT="production"

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

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_warning "This script is designed for Linux systems"
    fi
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    local available_space=$(df / | awk 'NR==2 {print $4}')
    local required_space=10485760  # 10GB in KB
    
    if [ $available_space -lt $required_space ]; then
        log_error "Insufficient disk space. Required: 10GB, Available: $(df -h / | awk 'NR==2 {print $4}')"
        exit 1
    fi
    
    # Check available memory (minimum 4GB)
    local available_memory=$(free -m | awk 'NR==2{print $2}')
    local required_memory=4096  # 4GB in MB
    
    if [ $available_memory -lt $required_memory ]; then
        log_error "Insufficient memory. Required: 4GB, Available: ${available_memory}MB"
        exit 1
    fi
    
    log_success "System requirements check passed"
}

install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install required packages
    local packages=(
        "curl"
        "wget"
        "git"
        "python3"
        "python3-pip"
        "python3-venv"
        "nodejs"
        "npm"
        "docker.io"
        "docker-compose"
        "postgresql-client"
        "redis-tools"
        "htop"
        "vim"
        "unzip"
        "jq"
        "mailutils"
        "logrotate"
        "cron"
        "ufw"
        "fail2ban"
        "certbot"
        "nginx"
    )
    
    for package in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            log_info "Installing $package..."
            sudo apt-get install -y "$package"
        else
            log_info "$package is already installed"
        fi
    done
    
    log_success "System dependencies installed"
}

setup_docker() {
    log_info "Setting up Docker..."
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Enable and start Docker service
    sudo systemctl enable docker
    sudo systemctl start docker
    
    # Configure Docker daemon
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "live-restore": true
}
EOF
    
    # Restart Docker
    sudo systemctl restart docker
    
    log_success "Docker setup completed"
}

setup_nginx() {
    log_info "Setting up Nginx..."
    
    # Create Nginx configuration
    sudo tee /etc/nginx/sites-available/$PROJECT_NAME > /dev/null <<EOF
upstream backend {
    server 127.0.0.1:8000;
}

upstream frontend {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Backend health check
    location /health {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Backend docs
    location /docs {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files
    location /static/ {
        alias /var/www/$PROJECT_NAME/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Logs
    access_log /var/log/nginx/$PROJECT_NAME.access.log;
    error_log /var/log/nginx/$PROJECT_NAME.error.log;
}
EOF
    
    # Enable site
    sudo ln -sf /etc/nginx/sites-available/$PROJECT_NAME /etc/nginx/sites-enabled/
    
    # Remove default site
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Test configuration
    sudo nginx -t
    
    # Start and enable Nginx
    sudo systemctl enable nginx
    sudo systemctl start nginx
    
    log_success "Nginx setup completed"
}

setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    # Check if domain is provided
    if [ -z "${DOMAIN:-}" ]; then
        log_warning "No domain provided. SSL setup skipped."
        log_warning "To set up SSL later, run: sudo certbot --nginx -d yourdomain.com"
        return 0
    fi
    
    # Install SSL certificate
    sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email "${ADMIN_EMAIL:-admin@$DOMAIN}"
    
    # Set up automatic renewal
    sudo systemctl enable certbot.timer
    sudo systemctl start certbot.timer
    
    log_success "SSL setup completed for domain: $DOMAIN"
}

setup_firewall() {
    log_info "Setting up firewall..."
    
    # Enable UFW
    sudo ufw --force enable
    
    # Allow SSH
    sudo ufw allow ssh
    
    # Allow HTTP and HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    
    # Allow Docker ports (for development)
    sudo ufw allow 8000/tcp
    sudo ufw allow 3000/tcp
    
    # Show status
    sudo ufw status
    
    log_success "Firewall setup completed"
}

setup_fail2ban() {
    log_info "Setting up Fail2ban..."
    
    # Create jail configuration
    sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3
EOF
    
    # Start and enable Fail2ban
    sudo systemctl enable fail2ban
    sudo systemctl start fail2ban
    
    log_success "Fail2ban setup completed"
}

setup_logging() {
    log_info "Setting up logging..."
    
    # Create log directories
    sudo mkdir -p /var/log/$PROJECT_NAME
    sudo mkdir -p /var/log/nginx
    sudo mkdir -p logs/backend
    sudo mkdir -p logs/frontend
    
    # Set permissions
    sudo chown -R $USER:$USER /var/log/$PROJECT_NAME
    sudo chown -R $USER:$USER logs
    
    # Configure logrotate
    sudo tee /etc/logrotate.d/$PROJECT_NAME > /dev/null <<EOF
/var/log/$PROJECT_NAME/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload nginx
    endscript
}

logs/backend/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}

logs/frontend/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}
EOF
    
    log_success "Logging setup completed"
}

setup_cron_jobs() {
    log_info "Setting up cron jobs..."
    
    # Create cron job for monitoring
    (crontab -l 2>/dev/null; echo "*/5 * * * * $(pwd)/backend/scripts/monitor.sh check >> /var/log/$PROJECT_NAME/monitor.log 2>&1") | crontab -
    
    # Create cron job for backups
    (crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/backend/scripts/backup.sh >> /var/log/$PROJECT_NAME/backup.log 2>&1") | crontab -
    
    # Create cron job for log cleanup
    (crontab -l 2>/dev/null; echo "0 3 * * 0 find /var/log/$PROJECT_NAME -name '*.log' -mtime +30 -delete") | crontab -
    
    log_success "Cron jobs setup completed"
}

setup_environment_variables() {
    log_info "Setting up environment variables..."
    
    # Create environment file
    cat > .env.production <<EOF
# Production Environment Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://ai_agent_user:$(openssl rand -base64 32)@localhost:5432/ai_agent_platform
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_agent_platform
DB_USER=ai_agent_user
DB_PASSWORD=$(openssl rand -base64 32)

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Security
SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 32)
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Configuration
API_V1_STR=/api/v1
PROJECT_NAME=AI Agent Platform
VERSION=1.0.0
DESCRIPTION=Production-ready AI Agent Platform

# CORS Configuration
BACKEND_CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]

# File Upload Configuration
MAX_FILE_SIZE=10485760
UPLOAD_DIR=uploads

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# Email Configuration
SMTP_TLS=true
SMTP_PORT=587
SMTP_HOST=smtp.gmail.com
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAILS_FROM_EMAIL=your-email@gmail.com
EMAILS_FROM_NAME=AI Agent Platform

# External Services
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment

# Backup Configuration
BACKUP_DIR=/var/backups/$PROJECT_NAME
BACKUP_RETENTION_DAYS=30

# Alert Configuration
ALERT_EMAIL=admin@yourdomain.com
EOF
    
    log_success "Environment variables setup completed"
    log_warning "Please update .env.production with your actual values"
}

setup_database() {
    log_info "Setting up database..."
    
    # Create database user
    sudo -u postgres psql <<EOF
CREATE USER ai_agent_user WITH PASSWORD '$(openssl rand -base64 32)';
CREATE DATABASE ai_agent_platform OWNER ai_agent_user;
GRANT ALL PRIVILEGES ON DATABASE ai_agent_platform TO ai_agent_user;
\q
EOF
    
    log_success "Database setup completed"
}

create_systemd_services() {
    log_info "Creating systemd services..."
    
    # Create service for the application
    sudo tee /etc/systemd/system/$PROJECT_NAME.service > /dev/null <<EOF
[Unit]
Description=AI Agent Platform
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0
User=$USER
Group=$USER

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable service
    sudo systemctl enable $PROJECT_NAME.service
    
    log_success "Systemd services created"
}

show_completion_info() {
    log_success "Production setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Update .env.production with your actual values"
    echo "2. Run database migrations: cd backend && python3 -m alembic upgrade head"
    echo "3. Start the application: sudo systemctl start $PROJECT_NAME"
    echo "4. Check status: sudo systemctl status $PROJECT_NAME"
    echo "5. View logs: docker-compose logs -f"
    echo ""
    echo "Useful commands:"
    echo "  Start application: sudo systemctl start $PROJECT_NAME"
    echo "  Stop application: sudo systemctl stop $PROJECT_NAME"
    echo "  Restart application: sudo systemctl restart $PROJECT_NAME"
    echo "  Check status: sudo systemctl status $PROJECT_NAME"
    echo "  View logs: docker-compose logs -f"
    echo "  Health check: ./backend/scripts/health_check.sh"
    echo "  Monitor: ./backend/scripts/monitor.sh start"
    echo "  Backup: ./backend/scripts/backup.sh"
    echo ""
    echo "Security notes:"
    echo "  - Firewall is enabled (UFW)"
    echo "  - Fail2ban is configured"
    echo "  - SSL certificates can be installed with: sudo certbot --nginx -d yourdomain.com"
    echo "  - Regular backups are scheduled"
    echo "  - Log rotation is configured"
    echo ""
}

# Main setup process
main() {
    log_info "Starting production setup for $PROJECT_NAME..."
    
    check_system_requirements
    install_system_dependencies
    setup_docker
    setup_nginx
    setup_ssl
    setup_firewall
    setup_fail2ban
    setup_logging
    setup_cron_jobs
    setup_environment_variables
    setup_database
    create_systemd_services
    show_completion_info
    
    log_success "Production setup completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "Usage: $0 [options]"
        echo ""
        echo "Environment Variables:"
        echo "  DOMAIN            Domain name for SSL setup"
        echo "  ADMIN_EMAIL       Admin email for SSL certificates"
        echo ""
        echo "Examples:"
        echo "  $0                                    # Basic setup"
        echo "  DOMAIN=example.com $0                # Setup with SSL"
        echo "  DOMAIN=example.com ADMIN_EMAIL=admin@example.com $0  # Full setup"
        exit 0
        ;;
    *)
        main
        ;;
esac
