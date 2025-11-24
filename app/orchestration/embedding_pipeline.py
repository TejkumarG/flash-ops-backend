"""
Embedding Pipeline Orchestrator: Generates embeddings from MongoDB tables.
"""
import time
from typing import Tuple

from app.services import get_mongo_client, get_vector_store
from app.models import EmbeddingGenerationResponse
from app.utils.logger import setup_logger

logger = setup_logger("embedding_pipeline")


class EmbeddingPipeline:
    """
    Orchestrator for embedding generation from MongoDB.
    """

    def __init__(self):
        """Initialize Embedding Pipeline."""
        logger.info("Initializing Embedding Pipeline...")
        self.mongo_client = get_mongo_client()
        self.vector_store = get_vector_store()
        logger.info("Embedding Pipeline initialized")

    def generate_embeddings(
        self,
        db_id: str,
        force_regenerate: bool = False
    ) -> EmbeddingGenerationResponse:
        """
        Generate embeddings for all tables in a database.

        Args:
            db_id: MongoDB database ID
            force_regenerate: Force regenerate even if embeddings exist

        Returns:
            EmbeddingGenerationResponse with status
        """
        start_time = time.time()
        logger.info(f"Generating embeddings for database ID: {db_id}")

        try:
            # Check if embeddings already exist
            if not force_regenerate and self.vector_store.loaded:
                stats = self.vector_store.get_stats()
                num_entities = stats.get('num_entities', 0)
                logger.info(f"Embeddings already synced for database. Collection has {num_entities} entities")
                return EmbeddingGenerationResponse(
                    status="success",
                    message=f"Database already synced. Collection '{self.vector_store.collection_name}' contains {num_entities} table embeddings. Use force_regenerate=true to override.",
                    tables_processed=num_entities,
                    embeddings_created=0,
                    index_path=f"milvus://{self.vector_store.collection_name}",
                    metadata_path=f"milvus://{self.vector_store.collection_name}",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )

            # Fetch tables from MongoDB (streaming generator)
            logger.info("Fetching tables from MongoDB (streaming)...")
            tables_generator = self.mongo_client.fetch_tables_for_database(db_id)

            # Process each batch as it arrives (no memory accumulation)
            total_tables_processed = 0
            total_vectors_created = 0
            chunk_num = 0

            for chunk in tables_generator:
                chunk_num += 1
                chunk_size = len(chunk)

                logger.info(f"Processing chunk {chunk_num}: {chunk_size} tables")

                try:
                    # Build index for this chunk only (delete existing on first chunk)
                    is_first_chunk = (chunk_num == 1)
                    collection_name, num_vectors = self.vector_store.build_index(
                        chunk,
                        db_id,
                        delete_existing=is_first_chunk
                    )

                    total_tables_processed += chunk_size
                    total_vectors_created += num_vectors

                    logger.info(f"Chunk {chunk_num} complete: {chunk_size} tables, {num_vectors} vectors (total: {total_tables_processed} tables, {total_vectors_created} vectors)")

                except Exception as e:
                    logger.error(f"Chunk {chunk_num} failed: {e}", exc_info=True)
                    logger.warning(f"Skipping chunk {chunk_num} and continuing with next chunk...")
                    # Continue processing remaining chunks even if this one failed
                    continue

            if total_tables_processed == 0:
                logger.error(f"No tables found for database {db_id}")
                return EmbeddingGenerationResponse(
                    status="error",
                    message=f"No tables found for database ID: {db_id}",
                    tables_processed=0,
                    embeddings_created=0,
                    index_path="",
                    metadata_path="",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )

            logger.info(f"All chunks processed: {total_tables_processed} tables, {total_vectors_created} vectors")

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Embeddings generated successfully in {processing_time}ms: "
                f"{total_tables_processed} tables processed"
            )

            return EmbeddingGenerationResponse(
                status="success",
                message="Embeddings generated successfully",
                tables_processed=total_tables_processed,
                embeddings_created=total_vectors_created,
                index_path=f"milvus://{collection_name}",
                metadata_path=f"milvus://{collection_name}",
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return EmbeddingGenerationResponse(
                status="error",
                message=f"Error generating embeddings: {str(e)}",
                tables_processed=0,
                embeddings_created=0,
                index_path="",
                metadata_path="",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )


# Global singleton instance
_embedding_pipeline = None


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Get Embedding Pipeline singleton."""
    global _embedding_pipeline
    if _embedding_pipeline is None:
        _embedding_pipeline = EmbeddingPipeline()
    return _embedding_pipeline
