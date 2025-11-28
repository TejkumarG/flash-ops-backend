"""
Response models for API endpoints.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class LoginResponse(BaseModel):
    """Response model for login."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: Dict[str, Any] = Field(..., description="User information")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "user": {
                    "id": "user_id",
                    "email": "user@example.com",
                    "name": "John Doe",
                    "role": "user"
                }
            }
        }


class QueryStatus(str, Enum):
    """Query execution status."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"


class JoinInfo(BaseModel):
    """Join information between tables."""
    from_table: str
    to_table: str
    condition: str
    type: str  # foreign_key, column_match, pattern_match


class QueryResponse(BaseModel):
    """Response model for query execution."""

    status: QueryStatus
    query: str
    tables_used: List[str]
    tier: int = Field(..., ge=1, le=3, description="Number of tables used (1-3)")
    row_count: int
    result: Optional[List[Dict[str, Any]]] = None
    csv_path: Optional[str] = None
    sql_generated: str
    joins: Optional[List[JoinInfo]] = None
    execution_time_ms: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    error_message: Optional[str] = None
    suggestions: Optional[List[str]] = None
    formatted_result: Optional[str] = Field(None, description="LLM-formatted result for display")
    file_path: Optional[str] = Field(None, description="MinIO parquet file path for large result sets")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "query": "show all active employees",
                "tables_used": ["employee_master"],
                "tier": 1,
                "row_count": 42,
                "result": [
                    {"emp_id": 111, "emp_name": "Remesh N", "status": 1}
                ],
                "sql_generated": "SELECT * FROM employee_master WHERE status = 1",
                "execution_time_ms": 2341,
                "confidence": 0.95
            }
        }


class EmbeddingGenerationResponse(BaseModel):
    """Response model for embedding generation."""

    status: str
    message: str
    tables_processed: int
    embeddings_created: int
    index_path: str
    metadata_path: str
    processing_time_ms: int

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Embeddings generated successfully",
                "tables_processed": 1500,
                "embeddings_created": 1500,
                "index_path": "data/embeddings/table_index.faiss",
                "metadata_path": "data/embeddings/table_metadata.pkl",
                "processing_time_ms": 45000
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    milvus_loaded: bool
    mongo_connected: bool
