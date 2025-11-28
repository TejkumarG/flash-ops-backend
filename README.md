# Flash-Ops: Natural Language to SQL System

Production-ready NL2SQL system supporting **1 crore (10 million) tables** with intelligent table selection and query generation.

## Features

- **Massive Scale**: Handles 1 crore tables using Milvus vector search
- **Intelligent Table Selection**: Clustering-based approach avoids redundant tables
- **Flexible Joins**: FK + column matching + pattern inference
- **Guaranteed Response**: 100% response rate with fallback chains
- **Clean Architecture**: PyTorch-style orchestration with modular agents
- **Docker Support**: Full containerization - one command to run everything

## Architecture

```
User Query
    |
[1] Vector Search (1 Crore -> Top 30 tables)      ~400ms
    |
[2] Table Clustering (30 -> Semantic groups)      ~15ms
    |
[3] Table Selector (Groups -> Best 1/2/3 tables)  ~250ms
    |
[4] Schema Packager (Collect metadata)            ~150ms
    |
[5] SQL Generator (LLM - temporary)               ~1200ms
    |
[6] Validator (Auto-repair)                       ~300ms
    |
[7] Executor (Run + format)                       ~200ms

Total: ~2.5 seconds per query
```

## Project Structure

```
flash-ops/
├── app/
│   ├── agents/              # Modular pipeline agents
│   │   ├── schema_scout.py           # Stage 1: Vector search
│   │   ├── table_clustering.py       # Stage 2: Clustering
│   │   ├── table_selector.py         # Stage 3: Table selection
│   │   ├── schema_packager.py        # Stage 4: Schema collection
│   │   ├── sql_generator.py          # Stage 5: SQL generation
│   │   └── quality_inspector.py      # Stage 6-7: Validation + execution
│   │
│   ├── orchestration/       # PyTorch-style handlers
│   │   ├── query_pipeline.py         # Main query orchestrator
│   │   └── embedding_pipeline.py     # Embedding generation orchestrator
│   │
│   ├── services/            # External integrations
│   │   ├── mongo_client.py           # MongoDB client
│   │   ├── vector_store.py           # Milvus vector store
│   │   └── minio_client.py           # MinIO storage client
│   │
│   ├── api/                 # Clean FastAPI routes
│   │   └── routes/
│   │       ├── query.py              # Query endpoint
│   │       ├── embeddings.py         # Embeddings endpoint
│   │       └── auth.py               # Authentication endpoint
│   │
│   ├── models/              # Pydantic models
│   │   ├── requests.py
│   │   └── responses.py
│   │
│   ├── config.py            # Configuration
│   └── main.py              # FastAPI app
│
├── data/
│   ├── exports/             # CSV exports
│   └── logs/                # Application logs
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── requirements.txt
```

## Quick Start

### Prerequisites
- Docker and Docker Compose installed ([Download Docker](https://www.docker.com/products/docker-desktop/))
- OpenAI API key

### Step 1: Setup Environment

```bash
# Clone and navigate to project
cd flash-ops

# Copy environment template
cp .env.example .env

# Edit .env file with your settings:
# - OPENAI_API_KEY=your_key_here
# - MONGO_DB_ID=your_database_id
```

### Step 2: Start All Services

```bash
# Start everything with Docker Compose
cd docker
docker-compose up -d

# Wait 60-90 seconds for all services to initialize
# Milvus takes some time to start up
```

This starts all services:
- **MongoDB**: localhost:27017 (database)
- **Milvus**: localhost:19530 (vector search)
- **MinIO**: localhost:9000, localhost:9001 (object storage)
- **FastAPI**: localhost:8000 (API server)
- **Attu**: localhost:3001 (Milvus web UI)

### Step 3: Verify Services

```bash
# Check health status
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"1.0.0","milvus_loaded":true,"mongo_connected":true}
```

### Step 4: Generate Embeddings & Test

```bash
# Generate embeddings for your database (first time only)
curl -X POST "http://localhost:8000/api/v1/embeddings/generate/default"

# Test with a query
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "show top 10 products"}'
```

### Managing Services

```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f fastapi
docker-compose logs -f milvus

# Stop all services
docker-compose down

# Stop and remove volumes (clean start)
docker-compose down -v
```

## Docker Services

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| MongoDB | `flash-ops-mongo` | 27017 | Database storage |
| FastAPI | `flash-ops-api` | 8000 | Main API server |
| Milvus | `milvus-standalone` | 19530 | Vector database |
| Attu | `milvus-attu` | 3001 | Milvus web UI |
| MinIO | `milvus-minio` | 9000, 9001 | Object storage |
| etcd | `milvus-etcd` | 2379 | Milvus metadata |

## Local Development (Without Docker)

For development without Docker:

```bash
# 1. Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env with your settings

# 4. Start external services with Docker
cd docker
docker-compose up -d mongodb milvus minio etcd

# 5. Run FastAPI locally
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### Query Processing

**POST** `/api/v1/query/`

```json
{
  "query": "emp id 111 from IT dept how much sales in May 2025"
}
```

**Response:**
```json
{
  "status": "success",
  "query": "...",
  "tables_used": ["employee_master", "sales_transaction", "department_master"],
  "tier": 3,
  "row_count": 1,
  "result": [{"total_sales": 8000.00}],
  "sql_generated": "SELECT SUM(s1.amount) as total_sales FROM...",
  "execution_time_ms": 2341,
  "confidence": 0.883
}
```

### Generate Embeddings

**POST** `/api/v1/embeddings/generate`

```json
{
  "db_id": "6919f70d1e144e4ea1b53ff4",
  "force_regenerate": false
}
```

### Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "milvus_loaded": true,
  "mongo_connected": true
}
```

## Configuration

Key settings in `app/config.py`:

```python
# Vector Search
VECTOR_SEARCH_TOP_K = 30              # Tables to retrieve
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer

# Clustering
CLUSTERING_SIMILARITY_THRESHOLD = 0.75  # Cluster threshold

# Table Selection
SINGLE_TABLE_SCORE_GAP = 0.2          # Gap to use single table
MAX_TABLES_PER_QUERY = 3              # Hard limit

# Results
MAX_RESULT_ROWS_IN_RESPONSE = 10      # JSON vs file threshold
```

## Visualization

Access Milvus data through **Attu Web UI**:
- Open http://localhost:3001
- Browse collections, view embeddings, and manage data

## Troubleshooting

### Services not starting
```bash
# Check service status
docker-compose ps

# View logs for specific service
docker-compose logs milvus
```

### Milvus connection timeout
Milvus takes 60-90 seconds to fully initialize. Wait and retry.

### MongoDB connection refused
Ensure MongoDB container is running:
```bash
docker-compose ps mongodb
```

### Port conflicts
If ports are in use, modify `docker-compose.yml` port mappings.

## License

MIT License
