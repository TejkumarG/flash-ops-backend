# Flash-Ops: Natural Language to SQL System

Production-ready NL2SQL system supporting **1 crore (10 million) tables** with intelligent table selection and query generation.

## 🚀 Features

- ✅ **Massive Scale**: Handles 1 crore tables using FAISS vector search
- ✅ **Intelligent Table Selection**: Clustering-based approach avoids redundant tables
- ✅ **Flexible Joins**: FK + column matching + pattern inference
- ✅ **Guaranteed Response**: 100% response rate with fallback chains
- ✅ **Clean Architecture**: PyTorch-style orchestration with modular agents
- ✅ **Docker Support**: Full containerization with visualization

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

## 🚦 Quick Start

### 1. Installation

```bash
# Clone repository
cd flash-ops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file:
```bash
# MongoDB connection
MONGO_URI=mongodb://localhost:27017
MONGO_DB_ID=6919f70d1e144e4ea1b53ff4

# OpenAI API Key (temporary)
OPENAI_API_KEY=your-key-here
```

### 3. Generate Embeddings

```bash
# Start the API server
uvicorn app.main:app --reload

# In another terminal, generate embeddings
curl -X POST "http://localhost:8000/api/v1/embeddings/generate/default"
```

### 4. Run Queries

```bash
# Query endpoint
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "show all active employees"}'
```

## 🐳 Docker Setup

```bash
# Start all services (API + MongoDB + Jupyter)
cd docker
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

**Services:**
- API: http://localhost:8000
- Jupyter: http://localhost:8888
- MongoDB: localhost:27017

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
  "faiss_index_loaded": true,
  "mongo_connected": true,
  "duckdb_connected": true
}
```

## 📈 Visualization

Open Jupyter notebook for interactive visualization:

```bash
# With Docker
docker-compose up jupyter
# Navigate to http://localhost:8888

# Or locally
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

**Built with ❤️ using FastAPI, FAISS, and clean architecture principles**
