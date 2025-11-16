"""API routes."""

from .query import router as query_router
from .embeddings import router as embeddings_router
from .auth import router as auth_router

__all__ = ["query_router", "embeddings_router", "auth_router"]
