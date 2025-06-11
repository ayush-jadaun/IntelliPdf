# IntelliPDF Deployment Guide

This guide covers deploying IntelliPDF to production environments, including cloud platforms, containerization, and scaling strategies.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Database Setup](#database-setup)
- [Monitoring & Logging](#monitoring--logging)
- [Security Configuration](#security-configuration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB minimum (16GB+ recommended)
- **Storage**: 100GB+ SSD (depends on document volume)
- **Network**: Stable internet connection for AI services

### Required Software
- Docker 24.0+ and Docker Compose 2.0+
- Python 3.11+ (if running without Docker)
- Node.js 18+ (if running without Docker)
- PostgreSQL 15+ (for production database)
- Redis 7+ (for caching and task queues)

### External Services
- **OpenAI API Key** (for GPT-4 integration)
- **Pinecone Account** (for vector database)
- **AWS S3** or **MinIO** (for file storage)

## Environment Configuration

### Production Environment Variables

Create a `.env.prod` file with the following configuration:

```bash
# Application
NODE_ENV=production
DEBUG=false
API_BASE_URL=https://api.yourdomain.com
FRONTEND_URL=https://yourdomain.com

# Database
DATABASE_URL=postgresql://user:password@db:5432/intellipdf
REDIS_URL=redis://redis:6379/0

# AI Services
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment

# File Storage
STORAGE_TYPE=s3  # or 'minio' or 'local'
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_BUCKET_NAME=intellipdf-documents
AWS_REGION=us-east-1

# Security
JWT_SECRET=your_super_secure_jwt_secret_here
CORS_ORIGINS=https://yourdomain.com
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com

# Performance
MAX_UPLOAD_SIZE=100MB
WORKER_PROCESSES=4
CELERY_WORKERS=8
VECTOR_DIMENSION=384
```

## Docker Deployment

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: intellipdf
      POSTGRES_USER: intellipdf_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    networks:
      - intellipdf-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - intellipdf-network

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    environment:
      - DATABASE_URL=postgresql://intellipdf_user:${DB_PASSWORD}@db:5432/intellipdf
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env.prod
    volumes:
      - ./data/uploads:/app/uploads
      - ./data/vector_db:/app/vector_db
    depends_on:
      - db
      - redis
    restart: unless-stopped
    networks:
      - intellipdf-network

  # Celery Worker
  celery-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    command: celery -A app.main worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql://intellipdf_user:${DB_PASSWORD}@db:5432/intellipdf
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env.prod
    volumes:
      - ./data/uploads:/app/uploads
      - ./data/vector_db:/app/vector_db
    depends_on:
      - db
      - redis
    restart: unless-stopped
    networks:
      - intellipdf-network

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    env_file:
      - .env.prod
    restart: unless-stopped
    networks:
      - intellipdf-network

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./deployment/nginx/ssl:/etc/nginx/ssl
      - ./data/uploads:/var/www/uploads
    depends_on:
      - backend
      - frontend
    restart: unless-stopped
    networks:
      - intellipdf-network

volumes:
  postgres_data:
  redis_data:

networks:
  intellipdf-network:
    driver: bridge
```

### Deployment Commands

```bash
# Clone repository
git clone https://github.com/your-org/intellipdf.git
cd intellipdf

# Set up environment
cp .env.example .env.prod
# Edit .env.prod with your production values

# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f backend
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS with Fargate

1. **Create ECS Cluster**:
```bash
aws ecs create-cluster --cluster-name intellipdf-cluster
```

2. **Build and Push Images**:
```bash
# Build images
docker build -t intellipdf-backend ./backend
docker build -t intellipdf-frontend ./frontend

# Tag for ECR
docker tag intellipdf-backend:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/intellipdf-backend:latest
docker tag intellipdf-frontend:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/intellipdf-frontend:latest

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/intellipdf-backend:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/intellipdf-frontend:latest
```

3. **Deploy using CloudFormation**:
Create `aws-infrastructure.yml` and deploy with:
```bash
aws cloudformation deploy --template-file aws-infrastructure.yml --stack-name intellipdf-stack --capabilities CAPABILITY_IAM
```

### Google Cloud Platform

#### Using Cloud Run

```bash
# Build and deploy backend
gcloud builds submit --tag gcr.io/your-project/intellipdf-backend ./backend
gcloud run deploy intellipdf-backend --image gcr.io/your-project/intellipdf-backend --platform managed --region us-central1

# Build and deploy frontend
gcloud builds submit --tag gcr.io/your-project/intellipdf-frontend ./frontend
gcloud run deploy intellipdf-frontend --image gcr.io/your-project/intellipdf-frontend --platform managed --region us-central1
```

### Azure Container Instances

```bash
# Create resource group
az group create --name intellipdf-rg --location eastus

# Deploy container group
az container create --resource-group intellipdf-rg --file azure-container-group.yml
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (v1.25+)
- kubectl configured
- Helm 3.0+ (optional)

### Deploy with Kubernetes

```bash
# Apply ConfigMaps and Secrets
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/secrets.yaml

# Deploy database
kubectl apply -f deployment/k8s/postgres-deployment.yaml
kubectl apply -f deployment/k8s/redis-deployment.yaml

# Deploy application
kubectl apply -f deployment/k8s/backend-deployment.yaml
kubectl apply -f deployment/k8s/frontend-deployment.yaml

# Set up ingress
kubectl apply -f deployment/k8s/ingress.yaml

# Check deployment status
kubectl get pods
kubectl get services
```

### Helm Chart Deployment

```bash
# Install with Helm
helm install intellipdf ./deployment/helm/intellipdf-chart \
  --set image.tag=latest \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=intellipdf.yourdomain.com
```

## Database Setup

### PostgreSQL Production Setup

1. **Initialize Database**:
```sql
-- Create database and user
CREATE DATABASE intellipdf;
CREATE USER intellipdf_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE intellipdf TO intellipdf_user;

-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

2. **Run Migrations**:
```bash
# From backend directory
alembic upgrade head
```

3. **Create Indexes**:
```sql
-- Performance indexes
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_embeddings_document_id ON embeddings(document_id);
```

### Vector Database Setup

#### Pinecone Setup
```python
import pinecone

# Initialize Pinecone
pinecone.init(
    api_key="your-api-key",
    environment="your-environment"
)

# Create index
pinecone.create_index(
    "intellipdf-embeddings",
    dimension=384,
    metric="cosine"
)
```

## Monitoring & Logging

### Application Monitoring

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'intellipdf-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
```

#### Grafana Dashboard
Import the provided dashboard configuration:
```bash
curl -X POST http://admin:admin@grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @deployment/monitoring/grafana-dashboard.json
```

### Centralized Logging

#### ELK Stack Setup
```yaml
# docker-compose.elk.yml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: logstash:8.8.0
    volumes:
      - ./deployment/logstash/logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: kibana:8.8.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
```

## Security Configuration

### SSL/TLS Setup

#### Let's Encrypt with Certbot
```bash
# Install certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Generate certificates
sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Security Headers

Add to Nginx configuration:
```nginx
# Security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
```

### API Rate Limiting

```python
# In FastAPI app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/upload")
@limiter.limit("10/minute")
async def upload_document(request: Request):
    # Upload logic
    pass
```

## Performance Optimization

### Backend Optimization

1. **Database Connection Pooling**:
```python
# SQLAlchemy configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

2. **Caching Strategy**:
```python
# Redis caching
@lru_cache(maxsize=1000)
async def get_document_embeddings(document_id: str):
    # Cache document embeddings
    pass
```

3. **Async Processing**:
```python
# Celery task for heavy processing
@celery_app.task
def process_document_async(document_id: str):
    # Background document processing
    pass
```

### Frontend Optimization

1. **Build Optimization**:
```javascript
// next.config.js
module.exports = {
  experimental: {
    optimizeCss: true,
  },
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },
  images: {
    domains: ['your-cdn-domain.com'],
  },
}
```

2. **CDN Configuration**:
```nginx
# Nginx CDN headers
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## Troubleshooting

### Common Issues

#### 1. High Memory Usage
```bash
# Check memory usage
docker stats

# Optimize Python memory
export PYTHONHASHSEED=0
export PYTHONOPTIMIZE=1
```

#### 2. Slow Vector Search
```python
# Optimize vector search
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.nprobe = 10  # Adjust for speed vs accuracy
```

#### 3. Database Connection Issues
```bash
# Check PostgreSQL connections
SELECT count(*) FROM pg_stat_activity;

# Optimize connection pool
# In .env
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
```

### Health Checks

#### Backend Health Check
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }
```

#### Database Health Check
```python
@app.get("/health/db")
async def db_health_check():
    try:
        # Test database connection
        result = await database.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Monitoring Commands

```bash
# System monitoring
htop
iotop
nethogs

# Docker monitoring
docker system df
docker container stats

# Database monitoring
psql -c "SELECT * FROM pg_stat_activity;"
redis-cli INFO memory
```

## Backup and Recovery

### Database Backup
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U intellipdf_user intellipdf > backup_$DATE.sql

# Upload to S3
aws s3 cp backup_$DATE.sql s3://intellipdf-backups/db/
```

### File Storage Backup
```bash
# Sync uploads to S3
aws s3 sync ./data/uploads s3://intellipdf-backups/uploads/
```

### Recovery Procedure
```bash
# Restore database
psql -h localhost -U intellipdf_user intellipdf < backup_20240101_120000.sql

# Restore files
aws s3 sync s3://intellipdf-backups/uploads/ ./data/uploads/
```

## Scaling Strategies

### Horizontal Scaling
- Use load balancers (Nginx, HAProxy, or cloud load balancers)
- Deploy multiple backend instances
- Implement session storage in Redis
- Use CDN for static assets

### Vertical Scaling
- Increase CPU and memory resources
- Optimize database queries
- Implement caching strategies
- Use async processing for heavy tasks

### Auto-scaling
- Configure Kubernetes HPA (Horizontal Pod Autoscaler)
- Set up cloud auto-scaling groups
- Monitor key metrics (CPU, memory, queue length)

---

For additional support or questions about deployment, please refer to our [API documentation](API.md) or contact the development team.