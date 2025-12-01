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
            # Check if this is an incremental sync (only tables with needs_sync=True)
            if not force_regenerate and self.vector_store.loaded:
                # Query Milvus for tables that need syncing
                tables_needing_sync = self.vector_store.get_tables_needing_sync(db_id)

                if not tables_needing_sync:
                    # No tables need syncing
                    stats = self.vector_store.get_stats()
                    num_entities = stats.get('num_entities', 0)
                    logger.info(f"No tables need syncing. Collection has {num_entities} entities")
                    return EmbeddingGenerationResponse(
                        status="success",
                        message=f"Database already synced. No tables with needs_sync=true. Collection contains {num_entities} table embeddings. Use force_regenerate=true to regenerate all.",
                        tables_processed=0,
                        embeddings_created=0,
                        index_path=f"milvus://{self.vector_store.collection_name}",
                        metadata_path=f"milvus://{self.vector_store.collection_name}",
                        processing_time_ms=int((time.time() - start_time) * 1000)
                    )

                # Sync only tables that need it
                logger.info(f"=" * 80)
                logger.info(f"INCREMENTAL SYNC: Found {len(tables_needing_sync)} tables with needs_sync=true")
                logger.info(f"=" * 80)

                # Use existing data from Milvus (edited in UI) - NO re-fetching from MSSQL
                # Now using separate description and field_descriptions fields
                import json
                tables_to_sync = []
                for table_entry in tables_needing_sync:
                    table_name = table_entry['table_name']
                    description = table_entry.get('description', '')
                    field_descriptions_json = table_entry.get('field_descriptions', '[]')
                    schema_json = table_entry.get('schema', '[]')
                    skipped = table_entry.get('skipped', False)

                    # Parse field_descriptions from JSON
                    try:
                        field_descriptions = json.loads(field_descriptions_json) if field_descriptions_json else []
                    except json.JSONDecodeError:
                        field_descriptions = []

                    # Parse columns from schema JSON
                    try:
                        columns = json.loads(schema_json) if schema_json else []
                    except json.JSONDecodeError:
                        columns = []

                    # Create table metadata with all fields
                    table_metadata = {
                        'table_name': table_name,
                        'description': description,
                        'field_descriptions': field_descriptions,
                        'columns': columns,
                        'skipped': skipped
                    }
                    tables_to_sync.append(table_metadata)

                    # Log field descriptions count
                    field_desc_count = len(field_descriptions) if field_descriptions else 0
                    logger.info(f"  - {table_name}: {description[:60] if description else 'No description'}{'...' if description and len(description) > 60 else ''} ({field_desc_count} field descriptions)")

                logger.info(f"\nUsing existing data from Milvus for {len(tables_to_sync)} tables (no MSSQL re-fetch)")

                # Write UI-edited descriptions to MSSQL for persistence
                logger.info(f"\n{'=' * 80}")
                logger.info(f"STEP 1: Writing UI-edited descriptions to MSSQL...")
                logger.info(f"{'=' * 80}")
                try:
                    from app.services.schema_extractor import SchemaExtractor

                    # Get database connection config
                    config_data = self.mongo_client.get_database_connection_config(db_id)
                    database_name = config_data["database_name"]
                    connection_config = config_data["connection_config"]

                    logger.info(f"Database: {database_name}")

                    # Create schema extractor
                    extractor = SchemaExtractor(connection_config)

                    # Write table descriptions in batch
                    logger.info(f"Writing table descriptions...")
                    num_added, num_updated, errors = extractor.write_table_descriptions_batch(
                        database_name,
                        tables_to_sync
                    )

                    if errors:
                        logger.warning(f"\n⚠ Some table descriptions failed to write: {len(errors)} errors")
                        for error in errors:
                            logger.warning(f"  - {error}")

                    logger.info(f"✓ Table descriptions: {num_added} added, {num_updated} updated, {len(errors)} errors")

                    # Write field descriptions in batch
                    logger.info(f"Writing field descriptions...")
                    field_num_added, field_num_updated, field_errors = extractor.write_field_descriptions_batch(
                        database_name,
                        tables_to_sync
                    )

                    if field_errors:
                        logger.warning(f"\n⚠ Some field descriptions failed to write: {len(field_errors)} errors")
                        for error in field_errors[:5]:  # Log first 5 errors
                            logger.warning(f"  - {error}")
                        if len(field_errors) > 5:
                            logger.warning(f"  ... and {len(field_errors) - 5} more errors")

                    logger.info(f"✓ Field descriptions: {field_num_added} added, {field_num_updated} updated, {len(field_errors)} errors")

                    logger.info(f"\n✓ MSSQL write complete")

                    extractor.close()

                except Exception as e:
                    # Log warning but continue with re-embedding (MSSQL write is optional)
                    logger.error(f"\n✗ Failed to write descriptions to MSSQL: {e}", exc_info=True)
                    logger.info("Continuing with re-embedding despite MSSQL write failure")

                # Upsert (update or insert) only these tables
                logger.info(f"\n{'=' * 80}")
                logger.info(f"STEP 2: Re-embedding {len(tables_to_sync)} tables with edited text from Milvus...")
                logger.info(f"{'=' * 80}")
                collection_name, num_vectors = self.vector_store.upsert_embeddings(
                    tables_to_sync,
                    db_id
                )

                processing_time = int((time.time() - start_time) * 1000)
                logger.info(f"\n✓ Incremental sync complete: {len(tables_to_sync)} tables re-embedded in {processing_time}ms")
                logger.info(f"{'=' * 80}")

                # Update MongoDB syncStatus to "synced"
                try:
                    self.mongo_client.update_sync_status(db_id, "synced")
                    logger.info(f"Updated MongoDB syncStatus to 'synced' for database {db_id}")
                except Exception as e:
                    logger.warning(f"Failed to update MongoDB syncStatus: {e}")

                return EmbeddingGenerationResponse(
                    status="success",
                    message=f"Incremental sync complete. Re-embedded {len(tables_to_sync)} tables that had needs_sync=true.",
                    tables_processed=len(tables_to_sync),
                    embeddings_created=num_vectors,
                    index_path=f"milvus://{collection_name}",
                    metadata_path=f"milvus://{collection_name}",
                    processing_time_ms=processing_time
                )

            # Full sync (force_regenerate=True or collection not loaded)
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

            # Update MongoDB syncStatus to "synced"
            try:
                self.mongo_client.update_sync_status(db_id, "synced")
                logger.info(f"Updated MongoDB syncStatus to 'synced' for database {db_id}")
            except Exception as e:
                logger.warning(f"Failed to update MongoDB syncStatus: {e}")

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
