# Flash-Ops: Natural Language to SQL System

Production-ready NL2SQL system supporting **1 crore (10 million) tables** with intelligent table selection and query generation.

## 🚀 Features

- ✅ **Massive Scale**: Handles 1 crore tables using Milvus vector search
- ✅ **Intelligent Table Selection**: Clustering-based approach avoids redundant tables
- ✅ **Flexible Joins**: FK + column matching + pattern inference
- ✅ **Guaranteed Response**: 100% response rate with fallback chains
- ✅ **Clean Architecture**: PyTorch-style orchestration with modular agents
- ✅ **Docker Support**: Full containerization - one command to run everything
- ✅ **Non-Programmer Friendly**: Simple setup, no complex configuration needed

## 📋 Architecture

```
User Query
    ↓
[1] Vector Search (1 Crore → Top 30 tables)      ~400ms
    ↓
[2] Table Clustering (30 → Semantic groups)      ~15ms
    ↓
[3] Table Selector (Groups → Best 1/2/3 tables)  ~250ms
    ↓
[4] Schema Packager (Collect metadata)           ~150ms
    ↓
[5] SQL Generator (LLM - temporary)              ~1200ms
    ↓
[6] Validator (Auto-repair)                      ~300ms
    ↓
[7] Executor (Run + format)                      ~200ms

Total: ~2.5 seconds per query
```

## 🏗️ Project Structure

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
│   │   ├── vector_store.py           # FAISS operations
│   │   └── db_client.py              # DuckDB client
│   │
│   ├── api/                 # Clean FastAPI routes
│   │   └── routes/
│   │       ├── query.py              # Query endpoint
│   │       └── embeddings.py         # Embeddings endpoint
│   │
│   ├── models/              # Pydantic models
│   │   ├── requests.py
│   │   └── responses.py
│   │
│   ├── config.py            # Configuration
│   └── main.py              # FastAPI app
│
├── data/
│   ├── embeddings/          # FAISS index + metadata
│   ├── exports/             # CSV exports
│   └── logs/                # Application logs
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── notebooks/
│   └── visualize.ipynb      # Visualization tools
│
└── requirements.txt
```

## 🚦 Quick Start (3 Steps - For Everyone!)

### Prerequisites
- Docker installed on your system ([Download Docker](https://www.docker.com/products/docker-desktop/))
- MongoDB running (either locally or in your Next.js app)
- OpenAI API key

### Step 1: Setup Environment

```bash
# 1. Navigate to project folder
cd flash-ops

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env file with your settings
# - Set MONGO_URI to your MongoDB connection (default: mongodb://localhost:27017)
# - Set MONGO_DB_ID to your database ID
# - Set OPENAI_API_KEY to your OpenAI key
```

**Important:** If MongoDB is on your host machine, use this in `.env`:
```bash
# For MongoDB running on host machine (outside Docker)
MONGO_URI=mongodb://host.docker.internal:27017
```

### Step 2: Start Everything (One Command!)

```bash
# Start all services with Docker
cd docker
docker-compose up -d

# Wait 30-60 seconds for services to be ready
```

That's it! All services are now running:
- ✅ **API Server**: http://localhost:8000
- ✅ **Attu UI** (Milvus Admin): http://localhost:3001
- ✅ **MinIO** (Storage): http://localhost:9001
- ✅ **Milvus** (Vector DB): Ready internally

### Step 3: Generate Embeddings & Test

```bash
# Generate embeddings for your database (first time only)
curl -X POST "http://localhost:8000/api/v1/embeddings/generate/default"

# Test with a query
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "show top 10 products"}'
```

### View Logs (Optional)

```bash
# View all logs
docker-compose logs -f

# View API logs only
docker-compose logs -f fastapi

# Stop everything
docker-compose down
```

## 🐳 Docker Services

The docker-compose setup includes:

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| FastAPI | `flash-ops-api` | 8000 | Main API server |
| Milvus | `milvus-standalone` | 19530 | Vector database |
| Attu | `milvus-attu` | 3001 | Milvus web UI |
| MinIO | `milvus-minio` | 9000, 9001 | Object storage |
| etcd | `milvus-etcd` | 2379 | Milvus metadata |

**Note:** MongoDB is NOT included in Docker - use your existing MongoDB instance.

## 💻 Local Development (For Developers)

If you prefer running without Docker:

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env with your settings

# 4. Start services manually
# - Start MongoDB (if not running)
# - Start Milvus with: docker-compose up -d milvus minio etcd attu

# 5. Run FastAPI
uvicorn app.main:app --reload --port 8000
```

## 📊 API Endpoints

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

## 📈 Visualization

You can access Milvus data through:

**Attu Web UI** (Recommended):
- Open http://localhost:3001
- Browse collections, view embeddings, and manage data
- No setup required - included in Docker

**Jupyter Notebooks** (Optional):
```bash
# Run locally
jupyter notebook notebooks/visualize.ipynb
```

**Visualizations include:**
- Table embeddings (PCA/t-SNE)
- Similarity heatmaps
- Clustering results
- Query performance metrics

## ⚙️ Configuration

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
MAX_RESULT_ROWS_IN_RESPONSE = 10      # JSON vs CSV threshold
```

## 🧪 Testing

```bash
# Run sample queries
python -m pytest tests/

# Or test via API
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "count all active users"}'
```

## 📝 Logging

Logs are written to `data/logs/` with component-specific files:
- `schema_scout_*.log`
- `table_clustering_*.log`
- `query_pipeline_*.log`

## 🔧 Development

### Adding New Agents

1. Create agent in `app/agents/`
2. Add to orchestration in `app/orchestration/query_pipeline.py`
3. Update factory functions

### Modifying Pipeline

Orchestrator uses PyTorch-style handler pattern:
```python
# app/orchestration/query_pipeline.py
def process(self, query: str):
    # Stage 1
    tables = self.schema_scout.search_tables(query)

    # Stage 2
    clusters = self.table_clustering.cluster_tables(tables)

    # Continue...
```

## 🚧 Roadmap

- [ ] **Phase 1**: Remove LLM dependency (replace with offline model)
- [ ] **Phase 2**: Add learning system (cache successful patterns)
- [ ] **Phase 3**: Improved query understanding
- [ ] **Phase 4**: Performance optimization (sub-2s response)

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request

## 📞 Support

For issues and questions:
- GitHub Issues: [Create issue](https://github.com/yourrepo/flash-ops/issues)
- Documentation: See `plan.md` for detailed architecture

---

**Built with ❤️ using FastAPI, Milvus, MongoDB, and clean architecture principles**
