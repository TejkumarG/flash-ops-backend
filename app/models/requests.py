"""
Request models for API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional


class LoginRequest(BaseModel):
    """Request model for user login."""

    email: str = Field(
        ...,
        description="User email address"
    )

    password: str = Field(
        ...,
        description="User password",
        min_length=6
    )

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "password123"
            }
        }


class QueryRequest(BaseModel):
    """Request model for natural language query."""

    query: str = Field(
        ...,
        description="Natural language query",
        min_length=3,
        max_length=500,
        examples=["emp id 111 from IT dept how much sales in May 2025"]
    )

    database_ids: list[str] = Field(
        ...,
        description="List of MongoDB database IDs to query against (frontend handles auth)",
        min_length=1,
        examples=[["6919f70d1e144e4ea1b53ff4", "691a15b01e144e4ea1b54023"]]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "show all active employees",
                "database_ids": ["6919f70d1e144e4ea1b53ff4"]
            }
        }


class EmbeddingGenerationRequest(BaseModel):
    """Request model for generating embeddings from MongoDB."""

    db_id: str = Field(
        ...,
        description="MongoDB database ID",
        examples=["6919f70d1e144e4ea1b53ff4"]
    )

    force_regenerate: bool = Field(
        default=False,
        description="Force regenerate embeddings even if they exist"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "db_id": "6919f70d1e144e4ea1b53ff4",
                "force_regenerate": False
            }
        }


class TeamApiKeyQueryRequest(BaseModel):
    """Request model for team API key query (runs query across all team databases)."""

    api_key: str = Field(
        ...,
        description="Team API key for authentication",
        min_length=32,
        examples=["a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"]
    )

    query: str = Field(
        ...,
        description="Natural language query",
        min_length=3,
        max_length=500,
        examples=["show all active employees"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                "query": "show total sales for each department"
            }
        }
