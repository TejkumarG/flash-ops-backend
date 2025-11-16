# Flash-Ops NL2SQL Architecture

## Overview

Flash-Ops is a production-grade Natural Language to SQL (NL2SQL) system designed for enterprise databases. The architecture follows **Single Responsibility Principle (SRP)** with clear separation of concerns across multiple specialized agents.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                      │
│  /api/v1/query/team, /api/v1/embeddings/*, /api/v1/databases/*  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Pipeline Orchestrator                   │
│  Coordinates all 7 stages of NL2SQL processing                  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐    ┌─────────────────┐
│  Stage 1-4    │     │   Stage 5        │    │   Stage 6-7     │
│  Table        │────▶│   SQL            │───▶│   Validation    │
│  Discovery    │     │   Generation     │    │   & Execution   │
└───────────────┘     └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐    ┌─────────────────┐
│  Milvus       │     │   OpenAI GPT     │    │   MSSQL/MySQL   │
│  Vector Store │     │   (LLM)          │    │   Target DB     │
└───────────────┘     └──────────────────┘    └─────────────────┘
```

## System Components

### 1. Data Layer

#### MongoDB (Metadata Storage)
- **Purpose**: Store database connection configs, API keys, team metadata
- **Collections**:
  - `databases`: Database connection configurations (encrypted credentials)
  - `teams`: Team API keys and permissions
- **Service**: `app/services/mongo_client.py`

#### Milvus (Vector Store)
- **Purpose**: Fast semantic search over table schemas (Stage 1 only)
- **Collection**: `table_embeddings`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Service**: `app/services/vector_store.py`
- **Note**: Only used for table discovery; rest of pipeline is vector-store agnostic

#### MinIO (Object Storage)
- **Purpose**: Store large query results as Parquet files
- **Bucket**: `query-results`
- **Service**: `app/services/minio_client.py`
- **Format**: Parquet (columnar, compressed)

#### Target Databases (MSSQL, MySQL, PostgreSQL - extensible)
- **Purpose**: Actual databases to query
- **Connection**: Dynamic via `SchemaExtractor`
- **Service**: `app/services/schema_extractor.py`

### 2. Agent Layer (7-Stage Pipeline)

#### Stage 1: Schema Scout (Vector Search)
- **File**: `app/agents/schema_scout.py`
- **Responsibility**: Find top-K relevant tables using semantic search
- **Input**: Natural language query + database_id
- **Output**: Top 30 tables with similarity scores
- **Dependencies**: Milvus, Sentence Transformers

#### Stage 2: Table Clustering
- **File**: `app/agents/table_clustering.py`
- **Responsibility**: Group similar tables to reduce redundancy
- **Algorithm**: Cosine similarity clustering (threshold: 0.75)
- **Output**: Clustered tables with representatives

#### Stage 3: Table Selector
- **File**: `app/agents/table_selector.py`
- **Responsibility**: Select optimal tables (1-3) and infer joins
- **Logic**:
  - Single table if score gap > 0.2
  - Multi-table with foreign key detection
  - Tier classification (1=single, 2=join, 3=complex)
- **Output**: Selected tables + join information

#### Stage 4: Schema Packager
- **File**: `app/agents/schema_packager.py`
- **Responsibility**: Format schema context for LLM
- **Output**: Markdown-formatted schema with columns, types, relationships

#### Stage 5: SQL Generator (with Reflection)
- **File**: `app/agents/sql_generator.py`
- **Responsibility**: Generate SQL from natural language
- **LLM**: OpenAI GPT-4o-mini
- **Reflection**: Up to 3 attempts with error feedback
- **Output**: Valid SQL query

#### Stage 6-7: Quality Inspector
- **File**: `app/agents/quality_inspector.py`
- **Responsibilities**:
  1. **Validation**: Dry-run SQL against target database
  2. **Execution**: Run query and fetch results
  3. **Auto-repair**: Fix common errors (e.g., 'active' → 1)
  4. **Formatting**: Generate LLM summaries
  5. **Storage**: Export large results to Parquet/MinIO
- **Output**: Results + metadata + natural language summary

### 3. Orchestration Layer

#### Query Pipeline
- **File**: `app/orchestration/query_pipeline.py`
- **Pattern**: PyTorch-style handler (sequential agent execution)
- **Flow**:
  ```
  process(query, database_id)
    → schema_scout.search_tables()
    → table_clustering.cluster_tables()
    → table_selector.select_tables()
    → schema_packager.package_schemas()
    → sql_generator.generate_sql() [with reflection]
    → quality_inspector.validate_and_execute()
    → quality_inspector.format_response()
    → return QueryResponse
  ```

#### Embedding Pipeline
- **File**: `app/orchestration/embedding_pipeline.py`
- **Purpose**: Index new databases into Milvus
- **Flow**:
  ```
  generate_embeddings(database_id)
    → schema_extractor.extract_schemas()
    → vector_store.build_index()
    → return stats
  ```

### 4. API Layer

#### Endpoints
```
POST /api/v1/query/team
  - Execute NL2SQL query with team API key
  - Input: {api_key, query}
  - Output: QueryResponse with results/file_path

POST /api/v1/embeddings/generate/{database_id}
  - Generate embeddings for database tables
  - Requires authentication

POST /api/v1/databases/register
  - Register new database connection
  - Encrypts credentials

GET /api/v1/databases/{database_id}
  - Get database metadata
```

## Data Flow

### Query Execution Flow

```
1. API Request
   ↓
2. Validate API Key (MongoDB lookup)
   ↓
3. Vector Search (Milvus) → Top 30 tables
   ↓
4. Clustering → Remove duplicates
   ↓
5. Selection → Choose 1-3 tables + joins
   ↓
6. Schema Packaging → Format for LLM
   ↓
7. SQL Generation (GPT-4) → 3 attempts with validation
   ↓
8. Execution (MSSQL/MySQL)
   ↓
9. Result Formatting:
   - If ≤10 rows: Return JSON + LLM summary
   - If >10 rows: Export Parquet to MinIO + LLM summary with top 5 rows
   ↓
10. Return QueryResponse
```

### Embedding Generation Flow

```
1. API Request (database_id)
   ↓
2. Fetch connection config (MongoDB)
   ↓
3. Connect to target database (MSSQL/MySQL)
   ↓
4. Extract schemas (INFORMATION_SCHEMA)
   ↓
5. Generate semantic descriptions
   ↓
6. Batch embedding (1000 tables/batch)
   ↓
7. Upsert to Milvus (delete old + insert new)
   ↓
8. Create IVF_FLAT index
   ↓
9. Return stats
```

## Design Principles

### 1. Single Responsibility Principle (SRP)
Each component has ONE job:
- **SchemaExtractor**: Database connections & schema extraction
- **QualityInspector**: SQL validation, execution, result formatting
- **MinIOClient**: Object storage operations
- **MongoClient**: Metadata persistence
- **MilvusVectorStore**: Vector search only
- **QueryPipeline**: Orchestration (delegates to agents)

### 2. Database Agnostic Design
- Connection type stored in MongoDB (`connectionType: 'mssql' | 'mysql' | 'postgresql'`)
- `SchemaExtractor` uses factory pattern for database-specific logic
- LLM prompts contain no database-specific syntax
- To add MySQL: implement `_extract_mysql_schemas()` in `SchemaExtractor`

### 3. Configuration-Driven
- All settings in `app/config.py` with `.env` override
- No hardcoded values (except defaults)
- Credentials encrypted with AES-256
- Environment-specific configs (dev/staging/prod)

### 4. Scalability
- Batch processing for embeddings (1000 tables/batch)
- Efficient Parquet storage for large results
- Connection pooling (future)
- Async operations (future enhancement)

### 5. Security
- API key authentication
- Encrypted database credentials
- Team-based access control
- Input validation and SQL injection prevention

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI | High-performance async web framework |
| **Metadata** | MongoDB | Flexible schema for configs |
| **Vector Store** | Milvus | Billion-scale vector search |
| **Object Storage** | MinIO | S3-compatible Parquet storage |
| **Embedding** | Sentence Transformers | Semantic table search |
| **LLM** | OpenAI GPT-4o-mini | SQL generation & result formatting |
| **Database** | MSSQL (pytds), MySQL* | Target databases |
| **Data Format** | Parquet | Efficient columnar storage |
| **Encryption** | Cryptography (Fernet) | AES-256 credential encryption |

*MySQL support requires implementation of `_extract_mysql_schemas()`

## Configuration

### Environment Variables (.env)
```bash
# Application
APP_NAME=Flash-Ops NL2SQL
DEBUG=True

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=flash-ops
MONGO_COLLECTION=databases

# Milvus (Vector Store)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=table_embeddings

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=1000

# Vector Search
VECTOR_SEARCH_TOP_K=30
CLUSTERING_SIMILARITY_THRESHOLD=0.75

# Table Selection
SINGLE_TABLE_SCORE_GAP=0.2
MAX_TABLES_PER_QUERY=3

# SQL Generation
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
MAX_REFLECTION_ATTEMPTS=3

# Results
MAX_RESULT_ROWS_IN_RESPONSE=10

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=query-results

# Encryption
ENCRYPTION_KEY=<32-byte-hex>
```

## Extension Points

### Adding a New Database Type

1. **Update `SchemaExtractor`** (`app/services/schema_extractor.py`):
```python
def extract_schemas(self, database_name: str):
    if self.connection_type == 'mssql':
        return self._extract_mssql_schemas(database_name)
    elif self.connection_type == 'mysql':
        return self._extract_mysql_schemas(database_name)  # NEW
    elif self.connection_type == 'postgresql':
        return self._extract_postgresql_schemas(database_name)  # NEW

def _extract_mysql_schemas(self, database_name: str):
    import mysql.connector
    # Implement MySQL-specific extraction using INFORMATION_SCHEMA
```

2. **Add validation logic**:
```python
def validate_sql(self, sql: str, database_name: str):
    if self.connection_type == 'mysql':
        return self._validate_mysql_sql(sql, database_name)
```

3. **Register database** with `connectionType: 'mysql'` in MongoDB

4. **Everything else works automatically!** (vector search, clustering, LLM, formatting)

### Adding a New Vector Store

Replace `MilvusVectorStore` with interface-compatible implementation:
- `connect()`, `build_index()`, `search()`, `disconnect()`
- Only affects Stage 1 (Schema Scout)

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Vector Search | 50-100ms | Top-30 from millions of tables |
| SQL Generation | 1-3s | 3 attempts with reflection |
| Query Execution | Variable | Depends on query complexity |
| Embedding Generation | ~1min/1000 tables | Batch processing |
| Result Formatting | 500ms-2s | LLM summary generation |

## Directory Structure

```
flash-ops/
├── app/
│   ├── agents/              # 7-stage pipeline agents
│   │   ├── schema_scout.py
│   │   ├── table_clustering.py
│   │   ├── table_selector.py
│   │   ├── schema_packager.py
│   │   ├── sql_generator.py
│   │   └── quality_inspector.py
│   ├── api/                 # FastAPI routes
│   │   ├── query.py
│   │   ├── embeddings.py
│   │   └── databases.py
│   ├── models/              # Pydantic models
│   │   ├── requests.py
│   │   └── responses.py
│   ├── orchestration/       # Pipeline orchestrators
│   │   ├── query_pipeline.py
│   │   └── embedding_pipeline.py
│   ├── services/            # External service clients
│   │   ├── mongo_client.py
│   │   ├── vector_store.py
│   │   ├── minio_client.py
│   │   └── schema_extractor.py
│   ├── utils/               # Utilities
│   │   ├── logger.py
│   │   ├── encryption.py
│   │   └── auth.py
│   ├── config.py            # Configuration
│   └── main.py              # FastAPI app
├── data/
│   └── logs/                # Application logs
├── docs/                    # Documentation
├── .env                     # Environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Future Enhancements

1. **Multi-Database Support**: PostgreSQL, BigQuery, Snowflake
2. **Query Caching**: Redis for frequent queries
3. **Async Operations**: Parallel agent execution
4. **Query Optimization**: Cost-based query planning
5. **Fine-tuned LLM**: Domain-specific SQL generation
6. **Web UI**: React dashboard for query management
7. **Monitoring**: Prometheus + Grafana metrics
8. **Multi-tenancy**: Namespace isolation

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
- [Sentence Transformers](https://www.sbert.net/)
- [Apache Parquet](https://parquet.apache.org/)
