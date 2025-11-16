"""
Embeddings API routes.
Clean routes that delegate to orchestrator.
"""
from fastapi import APIRouter, HTTPException

from app.models import EmbeddingGenerationRequest, EmbeddingGenerationResponse
from app.orchestration import get_embedding_pipeline
from app.config import settings

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.post("/generate", response_model=EmbeddingGenerationResponse)
async def generate_embeddings(
    request: EmbeddingGenerationRequest
) -> EmbeddingGenerationResponse:
    """
    Generate embeddings for tables from MongoDB.

    Args:
        request: Embedding generation request with database ID

    Returns:
        Embedding generation response with status
    """
    try:
        pipeline = get_embedding_pipeline()
        response = pipeline.generate_embeddings(
            db_id=request.db_id,
            force_regenerate=request.force_regenerate
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/default", response_model=EmbeddingGenerationResponse)
async def generate_default_embeddings() -> EmbeddingGenerationResponse:
    """
    Generate embeddings for default database from config.

    Returns:
        Embedding generation response with status
    """
    try:
        pipeline = get_embedding_pipeline()
        response = pipeline.generate_embeddings(
            db_id=settings.MONGO_DB_ID,
            force_regenerate=False
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
