# Enterprise Private Cloud Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying a private cloud instance of the AI Agent Platform for enterprise clients.

## Quick Start (Basic Deployment)

### Prerequisites
- Docker and Docker Compose installed
- SSL certificates for HTTPS (optional for internal deployment)
- Client-specific configuration details

### 1. Configuration Setup

```bash
# Copy the environment template
cp .env.template .env

# Edit the configuration for your client
nano .env
```

### 2. Required Environment Variables

```bash
# Minimum required configuration
CLIENT_NAME=AcmeCorp
CLIENT_DOMAIN=ai.acmecorp.com
CLIENT_DB_PASSWORD=secure_database_password_here
JWT_SECRET_KEY=generate_32_char_jwt_secret_key_here
ENCRYPTION_KEY=generate_32_char_encryption_key_here
GRAFANA_ADMIN_PASSWORD=secure_grafana_admin_password
```

### 3. Deploy the Stack

```bash
# Start the enterprise deployment
docker-compose -f docker-compose.enterprise.yml up -d

# Check deployment status
docker-compose -f docker-compose.enterprise.yml ps

# View logs
docker-compose -f docker-compose.enterprise.yml logs -f
```

### 4. Post-Deployment Setup

```bash
# Run database migrations
docker-compose -f docker-compose.enterprise.yml exec backend alembic upgrade head

# Create initial admin user (optional)
docker-compose -f docker-compose.enterprise.yml exec backend python scripts/create_admin_user.py

# Verify deployment
curl http://localhost:8000/api/v1/system/health
```

## Service Endpoints

- **API Backend**: `http://localhost:8000` (or your custom domain)
- **Grafana Monitoring**: `http://localhost:3000` (admin/[GRAFANA_ADMIN_PASSWORD])
- **Prometheus Metrics**: `http://localhost:9090`
- **Database**: `localhost:5432`
- **Redis**: `localhost:6379`

## Enterprise Features Included

### ✅ **Security & Compliance**
- Data isolation per client
- Encrypted data at rest and in transit
- Role-based access control
- Audit logging enabled
- JWT-based authentication

### ✅ **Monitoring & Observability**
- Prometheus metrics collection
- Grafana dashboards
- Application and infrastructure logs
- Health check endpoints
- Performance monitoring

### ✅ **Data Management**
- Automated database backups
- Data retention policies
- PostgreSQL with replication support
- Redis for session management

### ✅ **High Availability**
- Service restart policies
- Health checks for all services
- Load balancer ready (nginx included)
- Horizontal scaling support

## Configuration Options

### Database Scaling
```yaml
# In docker-compose.enterprise.yml
postgres:
  deploy:
    resources:
      limits:
        memory: 4G
        cpus: '2'
```

### API Scaling
```yaml
backend:
  deploy:
    replicas: 3
    resources:
      limits:
        memory: 2G
        cpus: '1'
```

### Enable SSO Integration
```bash
# In .env file
ENABLE_SSO=true
SSO_PROVIDER=saml
SSO_METADATA_URL=https://client.okta.com/app/metadata
```

## SSL/HTTPS Setup

### 1. Place SSL Certificates
```bash
mkdir -p nginx/ssl
# Copy your certificates
cp client.crt nginx/ssl/cert.pem
cp client.key nginx/ssl/key.pem
```

### 2. Update Nginx Configuration
```nginx
# nginx/nginx.conf
server {
    listen 443 ssl;
    server_name ai.client.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://backend:8000;
    }
}
```

## Backup & Recovery

### Automated Backups
```bash
# Database backup (runs daily at 2 AM)
docker-compose -f docker-compose.enterprise.yml exec postgres pg_dump -U postgres ai_agent_platform_client > backup/$(date +%Y%m%d).sql

# Manual backup
docker-compose -f docker-compose.enterprise.yml exec postgres pg_dump -U postgres ai_agent_platform_client | gzip > backup/manual_$(date +%Y%m%d_%H%M%S).sql.gz
```

### Recovery
```bash
# Restore from backup
gunzip -c backup/20231215_143000.sql.gz | docker-compose -f docker-compose.enterprise.yml exec -T postgres psql -U postgres ai_agent_platform_client
```

## Monitoring & Alerting

### Grafana Dashboards
- **Application Metrics**: Request rates, response times, error rates
- **Infrastructure**: CPU, memory, disk usage
- **Database**: Connection counts, query performance
- **Business Metrics**: Agent usage, conversation counts

### Prometheus Alerts
```yaml
# monitoring/alerts.yml
groups:
  - name: ai-platform
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: High error rate detected
```

## Scaling Recommendations

### Small Enterprise (< 1000 users)
```yaml
resources:
  postgres: 2 CPU, 8GB RAM, 100GB SSD
  redis: 1 CPU, 2GB RAM
  backend: 2 instances, 1 CPU, 2GB RAM each
```

### Medium Enterprise (1000-10000 users)
```yaml
resources:
  postgres: 4 CPU, 16GB RAM, 500GB SSD
  redis: 2 CPU, 4GB RAM
  backend: 4 instances, 2 CPU, 4GB RAM each
```

### Large Enterprise (> 10000 users)
```yaml
resources:
  postgres: 8 CPU, 32GB RAM, 1TB SSD
  redis: 4 CPU, 8GB RAM
  backend: 8 instances, 4 CPU, 8GB RAM each
```

## Security Hardening

### Network Security
```bash
# Restrict access to internal networks only
iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 8000 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

### Database Security
```sql
-- Create read-only user for monitoring
CREATE USER monitoring WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE ai_agent_platform_client TO monitoring;
GRANT USAGE ON SCHEMA public TO monitoring;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring;
```

## Troubleshooting

### Common Issues

**Backend not starting**:
```bash
# Check logs
docker-compose -f docker-compose.enterprise.yml logs backend

# Common fix: Database connection issues
docker-compose -f docker-compose.enterprise.yml restart postgres
```

**High memory usage**:
```bash
# Monitor resource usage
docker stats

# Adjust resource limits in docker-compose.enterprise.yml
```

**SSL certificate issues**:
```bash
# Verify certificate
openssl x509 -in nginx/ssl/cert.pem -text -noout

# Check nginx configuration
docker-compose -f docker-compose.enterprise.yml exec nginx nginx -t
```

## Support & Maintenance

### Regular Maintenance Tasks
- Weekly backup verification
- Monthly security updates
- Quarterly performance reviews
- Annual certificate renewal

### Support Contacts
- **Technical Issues**: tech-support@yourcompany.com
- **Security Concerns**: security@yourcompany.com
- **Emergency**: emergency-hotline@yourcompany.com

---

## Next Steps

1. **Advanced SSO**: Configure SAML/OIDC integration
2. **Custom Branding**: White-label the interface
3. **API Integrations**: Connect to client's existing systems
4. **Advanced Monitoring**: Set up alerts and dashboards
5. **Disaster Recovery**: Implement cross-region backups