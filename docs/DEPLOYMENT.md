# Deployment Guide

## Production Deployment for Flash-Ops NL2SQL

This guide covers deploying Flash-Ops to production environments.

## Prerequisites

- Docker & Docker Compose
- 4GB+ RAM
- 20GB+ disk space
- Python 3.11+

## Infrastructure Components

### Required Services

1. **MongoDB** - Metadata storage
2. **Milvus** - Vector database
3. **MinIO** - Object storage
4. **FastAPI App** - Main application
5. **Nginx** (optional) - Reverse proxy

## Quick Start with Docker Compose

### 1. Create `docker-compose.yml`

```yaml
version: '3.8'

services:
  # MongoDB
  mongodb:
    image: mongo:7.0
    container_name: flash-ops-mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
    networks:
      - flash-ops-network
    restart: unless-stopped

  # Milvus (Standalone)
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: flash-ops-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    networks:
      - flash-ops-network
    restart: unless-stopped

  minio:
    image: minio/minio:RELEASE-2024-01-16T16-07-38Z
    container_name: flash-ops-minio-milvus
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - flash-ops-network
    restart: unless-stopped

  milvus:
    image: milvusdb/milvus:v2.3.3
    container_name: flash-ops-milvus
    depends_on:
      - etcd
      - minio
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    command: ["milvus", "run", "standalone"]
    networks:
      - flash-ops-network
    restart: unless-stopped

  # MinIO (Object Storage for Query Results)
  minio-storage:
    image: minio/minio:latest
    container_name: flash-ops-minio-storage
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    volumes:
      - minio_storage_data:/data
    command: server /data --console-address ":9001"
    networks:
      - flash-ops-network
    restart: unless-stopped

  # Flash-Ops Application
  flash-ops:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: flash-ops-app
    depends_on:
      - mongodb
      - milvus
      - minio-storage
    ports:
      - "8000:8000"
    environment:
      - MONGO_URI=mongodb://admin:${MONGO_PASSWORD}@mongodb:27017
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - MINIO_ENDPOINT=minio-storage:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=${MINIO_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
    volumes:
      - ./data/logs:/app/data/logs
    networks:
      - flash-ops-network
    restart: unless-stopped

networks:
  flash-ops-network:
    driver: bridge

volumes:
  mongodb_data:
  etcd_data:
  minio_data:
  milvus_data:
  minio_storage_data:
```

### 2. Create `.env` File

```bash
# MongoDB
MONGO_PASSWORD=your_secure_password

# MinIO
MINIO_PASSWORD=your_secure_password

# OpenAI
OPENAI_API_KEY=sk-...

# Encryption Key (generate with: openssl rand -hex 32)
ENCRYPTION_KEY=your_32_byte_hex_key

# Application
DEBUG=False
```

### 3. Create `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY data/ data/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 4. Deploy

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f flash-ops

# Initialize MinIO bucket
docker exec flash-ops-minio-storage mc mb /data/query-results

# Verify services
curl http://localhost:8000/health
```

## Production Configuration

### Environment Variables

```bash
# Application
APP_NAME=Flash-Ops NL2SQL
DEBUG=False
LOG_LEVEL=INFO

# MongoDB
MONGO_URI=mongodb://admin:password@mongodb:27017
MONGO_DB_NAME=flash-ops
MONGO_COLLECTION=databases

# Milvus
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=table_embeddings

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=1000

# Vector Search
VECTOR_SEARCH_TOP_K=30
CLUSTERING_SIMILARITY_THRESHOLD=0.75

# SQL Generation
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
MAX_REFLECTION_ATTEMPTS=3

# Results
MAX_RESULT_ROWS_IN_RESPONSE=10

# MinIO
MINIO_ENDPOINT=minio-storage:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=your_password
MINIO_BUCKET=query-results
MINIO_SECURE=False

# Encryption
ENCRYPTION_KEY=your_32_byte_hex_key

# Workers
UVICORN_WORKERS=4
```

## Security Hardening

### 1. Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/flash-ops
upstream flash_ops {
    server localhost:8000;
}

server {
    listen 80;
    server_name api.flashops.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.flashops.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/api.flashops.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.flashops.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    location / {
        proxy_pass http://flash_ops;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 2. Firewall Rules

```bash
# Allow only necessary ports
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw enable
```

### 3. API Key Rotation

```python
# Rotate API keys quarterly
POST /api/v1/teams/{team_id}/rotate-key
```

## Monitoring

### 1. Health Check Endpoint

```python
# app/api/health.py
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "mongodb": check_mongodb(),
            "milvus": check_milvus(),
            "minio": check_minio()
        }
    }
```

### 2. Logging

```yaml
# docker-compose.yml - Add logging driver
services:
  flash-ops:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 3. Metrics (Future)

```python
# Add Prometheus metrics
pip install prometheus-fastapi-instrumentator

# app/main.py
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

## Backup Strategy

### 1. MongoDB Backup

```bash
# Daily backup cron
0 2 * * * docker exec flash-ops-mongodb mongodump --out /backup/$(date +\%Y\%m\%d)
```

### 2. Milvus Backup

```bash
# Backup Milvus data directory
tar -czf milvus-backup-$(date +%Y%m%d).tar.gz /var/lib/docker/volumes/milvus_data
```

### 3. MinIO Backup

```bash
# Mirror to S3
mc mirror minio-storage/query-results s3/backup-bucket/query-results
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  flash-ops:
    deploy:
      replicas: 3
    # Add load balancer
```

### Vertical Scaling

```yaml
services:
  flash-ops:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Troubleshooting

### Common Issues

1. **Milvus connection timeout**
   ```bash
   # Increase timeout
   MILVUS_CONNECTION_TIMEOUT=60
   ```

2. **MongoDB authentication failed**
   ```bash
   # Check credentials
   docker exec -it flash-ops-mongodb mongosh -u admin -p
   ```

3. **MinIO bucket not found**
   ```bash
   # Create bucket
   docker exec flash-ops-minio-storage mc mb /data/query-results
   ```

### Logs

```bash
# View application logs
docker-compose logs -f flash-ops

# View MongoDB logs
docker-compose logs -f mongodb

# View Milvus logs
docker-compose logs -f milvus
```

## Performance Tuning

### 1. Milvus Index Optimization

```python
# Use IVF_SQ8 for faster search with memory savings
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_SQ8",
    "params": {"nlist": 1024}
}
```

### 2. Connection Pooling

```python
# Add to config.py
CONNECTION_POOL_SIZE=10
```

### 3. Caching (Future)

```python
# Add Redis for query caching
pip install redis
```

## Cost Optimization

### Infrastructure Costs (AWS Example)

| Service | Instance Type | Monthly Cost |
|---------|--------------|--------------|
| EC2 (API) | t3.medium | $30 |
| DocumentDB | t3.medium | $70 |
| S3 (MinIO) | 100GB | $2 |
| **Total** | | **~$102/month** |

### Self-Hosted Alternative

| Service | Hardware | Monthly Cost |
|---------|----------|--------------|
| VPS | 8GB RAM, 4 vCPU | $40 |
| **Total** | | **~$40/month** |

## Maintenance

### Regular Tasks

1. **Weekly**
   - Review logs for errors
   - Monitor disk usage
   - Check API response times

2. **Monthly**
   - Update dependencies
   - Rotate API keys
   - Review security patches

3. **Quarterly**
   - Performance audit
   - Cost optimization review
   - Database cleanup

## Support

For production issues:
- GitHub Issues: https://github.com/yourusername/flash-ops/issues
- Email: support@flashops.com
