"""
FastAPI application for NL2SQL system.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api.routes import query_router, embeddings_router, auth_router
from app.models import HealthResponse
from app.services import get_mongo_client, get_vector_store
from app.utils.logger import setup_logger

logger = setup_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting up Flash-Ops NL2SQL...")

    # Initialize services
    logger.info("Initializing services...")
    mongo_client = get_mongo_client()
    vector_store = get_vector_store()

    logger.info("Services initialized successfully")
    logger.info(f"Milvus loaded: {vector_store.loaded}")
    logger.info(f"MongoDB connected: {mongo_client.connected}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    mongo_client.disconnect()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Natural Language to SQL System with 1 crore table support",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(query_router, prefix=settings.API_V1_PREFIX)
app.include_router(embeddings_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    mongo_client = get_mongo_client()
    vector_store = get_vector_store()

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        milvus_loaded=vector_store.loaded,
        mongo_connected=mongo_client.connected,
        duckdb_connected=False  # Legacy field, kept for API compatibility
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
