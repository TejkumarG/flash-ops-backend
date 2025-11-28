"""
Configuration settings for the Natural Language to SQL system.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "Flash-Ops NL2SQL"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_ID: str = "6919f70d1e144e4ea1b53ff4"
    MONGO_DB_NAME: str = "flash-ops"
    MONGO_COLLECTION: str = "databases"

    # Vector Search (Milvus)
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "table_embeddings"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    VECTOR_SEARCH_TOP_K: int = 30

    # Batching (for scalability with millions of tables)
    EMBEDDING_BATCH_SIZE: int = 1000  # Process tables in batches

    # Table Clustering
    CLUSTERING_SIMILARITY_THRESHOLD: float = 0.75

    # Table Selection
    SINGLE_TABLE_SCORE_GAP: float = 0.2
    MAX_TABLES_PER_QUERY: int = 3

    # SQL Generation (Temporary - LLM)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    MAX_REFLECTION_ATTEMPTS: int = 3
    TEMPERATURE_START: float = 0.1
    TEMPERATURE_INCREMENT: float = 0.05

    # Query Settings
    CONNECTION_POOL_SIZE: int = 3
    QUERY_TIMEOUT_SECONDS: int = 30

    # Results
    MAX_RESULT_ROWS_IN_RESPONSE: int = 10
    CSV_EXPORT_PATH: str = "data/exports/"
    RESULT_FORMATTER_THRESHOLD: int = 5  # Use LLM formatter for ≤5 records, MinIO for >5

    # MinIO (Object Storage)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "query-results"
    MINIO_SECURE: bool = False  # Use HTTPS

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_PATH: str = "data/logs/"

    # Encryption (for database credentials)
    ENCRYPTION_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
