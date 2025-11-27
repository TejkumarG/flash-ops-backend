"""
Milvus vector store for table embeddings.
"""
from typing import List, Dict, Any, Tuple
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("vector_store")


class MilvusVectorStore:
    """Milvus vector store for table embeddings."""

    def __init__(self):
        """Initialize Milvus vector store."""
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.model = None
        self.collection = None
        self.connected = False
        self.loaded = False

    def connect(self) -> bool:
        """
        Connect to Milvus server.

        Returns:
            True if connection successful
        """
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.connected = True
            logger.info(f"Connected to Milvus: {self.host}:{self.port}")

            # Load collection if exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                self.loaded = True
                logger.info(f"Loaded collection: {self.collection_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from Milvus."""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
            self.loaded = False
            logger.info("Disconnected from Milvus")

    def _generate_id(self, database_id: str, table_name: str) -> int:
        """
        Generate deterministic ID from database_id and table_name.

        Args:
            database_id: MongoDB database ID
            table_name: Table name

        Returns:
            Deterministic 64-bit integer ID
        """
        # Create unique string
        unique_str = f"{database_id}:{table_name}"
        # Hash it
        hash_bytes = hashlib.sha256(unique_str.encode()).digest()
        # Convert first 8 bytes to int64 (positive)
        return int.from_bytes(hash_bytes[:8], byteorder='big', signed=False) & 0x7FFFFFFFFFFFFFFF

    def _generate_semantic_description(self, table_name: str) -> str:
        """
        Generate semantic description from table name for better embeddings.

        Args:
            table_name: Name of the table

        Returns:
            Natural language description
        """
        # Convert PascalCase/camelCase/snake_case to words
        import re
        # Handle underscores
        table_name = table_name.replace('_', ' ')
        # Handle PascalCase/camelCase
        words = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', table_name)).split()
        base_words = ' '.join(words).lower().strip()

        # Generate simple semantic description
        return f"This table contains {base_words} data and related information"

    def _load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")

    def load_encoder(self):
        """
        Load encoder for table clustering.
        Alias for _load_model() with encoder attribute for backward compatibility.
        """
        self._load_model()
        self.encoder = self.model

    def build_index(self, tables: List[Dict[str, Any]], database_id: str, delete_existing: bool = True) -> Tuple[str, int]:
        """
        Build Milvus collection from tables with batching support.
        Only updates embeddings for the specific database_id (preserves other databases).

        Args:
            tables: List of table metadata
            database_id: MongoDB database ID to associate with tables
            delete_existing: Whether to delete existing embeddings for this database_id (default: True)

        Returns:
            Tuple of (collection_name, num_vectors)
        """
        if not self.connected:
            raise ConnectionError("Milvus not connected")

        try:
            # Load model
            self._load_model()

            # Check if collection exists
            collection_exists = utility.has_collection(self.collection_name)

            if collection_exists:
                # Collection exists
                self.collection = Collection(self.collection_name)

                # Delete existing embeddings for this database_id (only if requested)
                if delete_existing:
                    expr = f'database_id == "{database_id}"'
                    self.collection.delete(expr)
                    logger.info(f"Deleted existing embeddings for database_id: {database_id}")
            else:
                # Create new collection
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="database_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                    FieldSchema(name="schema", dtype=DataType.VARCHAR, max_length=8192),  # JSON schema of columns
                    FieldSchema(name="needs_sync", dtype=DataType.BOOL),
                    FieldSchema(name="skipped", dtype=DataType.BOOL),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
                ]
                schema = CollectionSchema(fields, description="Table embeddings for NL2SQL")

                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using='default'
                )
                logger.info(f"Created collection: {self.collection_name}")

            # Process tables in batches
            batch_size = settings.EMBEDDING_BATCH_SIZE
            total_tables = len(tables)
            total_batches = (total_tables + batch_size - 1) // batch_size
            total_inserted = 0

            logger.info(f"Processing {total_tables} tables in {total_batches} batches (batch_size={batch_size})")

            for batch_idx in range(0, total_tables, batch_size):
                batch_num = (batch_idx // batch_size) + 1
                batch_tables = tables[batch_idx:batch_idx + batch_size]
                batch_count = len(batch_tables)

                logger.info(f"Processing batch {batch_num}/{total_batches}: {batch_count} tables")

                # Generate embeddings for this batch
                texts = []
                for table in batch_tables:
                    table_name = table.get('table_name', '')

                    # Create semantic text for embedding (table name + description only)
                    # Columns are stored separately in the 'schema' field
                    text_parts = [f"Table: {table_name}"]

                    # Use user-provided description if available, otherwise generate one
                    user_description = table.get('description')
                    if user_description:
                        # Use user-provided description from MSSQL extended properties
                        text_parts.append(f"Description: {user_description}")
                        logger.debug(f"[DESC] {table_name}: Using user description: {user_description[:100]}")
                    else:
                        # Fall back to auto-generated description from table name
                        generated_desc = self._generate_semantic_description(table_name)
                        text_parts.append(f"Description: {generated_desc}")
                        logger.debug(f"[DESC] {table_name}: Using auto-generated description")

                    text = "\n".join(text_parts)
                    texts.append(text)

                # Log sample of embedding texts for debugging
                if batch_num == 1 and len(texts) > 0:
                    logger.info(f"[EMBEDDING TEXT SAMPLE] First table embedding text:\n{texts[0][:500]}")

                # Encode batch
                embeddings = self.model.encode(texts, show_progress_bar=False)
                logger.info(f"Batch {batch_num}/{total_batches}: Generated embeddings with shape {embeddings.shape}")

                # Generate IDs for this batch
                batch_ids = [
                    self._generate_id(database_id, t.get('table_name', ''))
                    for t in batch_tables
                ]

                # Prepare batch data - with schema
                import json
                batch_data = [
                    batch_ids,  # Deterministic IDs for updating records
                    [database_id] * batch_count,  # database_id for all tables in batch
                    [t.get('table_name', '') for t in batch_tables],
                    [text[:4096] for text in texts],  # Store the text used for embedding
                    [json.dumps(t.get('columns', []))[:8192] for t in batch_tables],  # Schema as JSON
                    [False] * batch_count,  # needs_sync - always False after sync completion
                    [False] * batch_count,  # skipped - always False initially (UI can control)
                    embeddings.tolist()
                ]

                # Insert batch
                self.collection.insert(batch_data)
                total_inserted += batch_count
                logger.info(f"Batch {batch_num}/{total_batches}: Inserted {batch_count} vectors (total: {total_inserted}/{total_tables})")

            logger.info(f"All batches processed: Inserted {total_inserted} vectors into Milvus")

            # Create index only if collection was just created (not if updating existing)
            if not collection_exists:
                logger.info("Creating index on embedding field...")
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                logger.info("Created index on embedding field")

            # Load/reload collection
            self.collection.load()
            self.loaded = True
            logger.info("Collection loaded and ready for search")

            return self.collection_name, total_inserted

        except Exception as e:
            logger.error(f"Error building Milvus index: {e}")
            raise

    def search(
        self,
        query: str,
        database_id: str,
        k: int = None,
        metric_type: str = "L2"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar tables using query, filtered by database_id and skipped status.

        Args:
            query: Natural language query
            database_id: MongoDB database ID to filter results
            k: Number of results to return (default: settings.VECTOR_SEARCH_TOP_K)
            metric_type: Distance metric (L2 or IP)

        Returns:
            List of similar tables with scores (excludes tables with skipped=true)
        """
        if not self.connected:
            raise ConnectionError("Milvus not connected")

        if not self.loaded or self.collection is None:
            raise ValueError("Collection not loaded")

        if k is None:
            k = settings.VECTOR_SEARCH_TOP_K

        try:
            # Load model if not loaded
            self._load_model()

            # Generate query embedding
            query_embedding = self.model.encode([query], show_progress_bar=False)

            # Search parameters
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 10}
            }

            # Perform search with database_id and skipped filter
            results = self.collection.search(
                data=query_embedding.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=k,
                expr=f'database_id == "{database_id}" && skipped == false',  # Filter by database_id and exclude skipped tables
                output_fields=["database_id", "table_name", "text", "schema"]
            )

            # Format results
            tables = []
            for hits in results:
                for hit in hits:
                    # Convert L2 distance to score (inverse distance)
                    distance = hit.distance
                    score = 1.0 / (1.0 + distance) if metric_type == "L2" else hit.distance

                    # Parse schema JSON back to Python object
                    import json
                    schema_json = hit.entity.get("schema") or "[]"
                    try:
                        columns = json.loads(schema_json) if schema_json else []
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse schema JSON for table {hit.entity.get('table_name')}")
                        columns = []

                    table = {
                        "database_id": hit.entity.get("database_id"),
                        "table_name": hit.entity.get("table_name"),
                        "text": hit.entity.get("text"),
                        "columns": columns,  # Include parsed columns
                        "score": score,
                        "distance": distance
                    }
                    tables.append(table)

            logger.info(f"Vector search returned {len(tables)} results for database {database_id}, query: {query[:50]}...")
            return tables

        except Exception as e:
            logger.error(f"Error searching Milvus: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        if not self.connected or self.collection is None:
            return {
                "connected": False,
                "collection_name": self.collection_name,
                "num_entities": 0
            }

        try:
            stats = {
                "connected": True,
                "collection_name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "dimension": self.dimension
            }
            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "connected": False,
                "error": str(e)
            }

    def upsert_embeddings(
        self,
        tables: List[Dict[str, Any]],
        database_id: str
    ) -> Tuple[str, int]:
        """
        Upsert embeddings for tables (update if exists, insert if not).
        Uses deterministic IDs for upsert functionality.

        Args:
            tables: List of table metadata
            database_id: MongoDB database ID

        Returns:
            Tuple of (collection_name, num_vectors)
        """
        if not self.connected or not self.loaded:
            raise ConnectionError("Milvus not connected or collection not loaded")

        try:
            # Load model
            self._load_model()

            # Process tables in batches
            batch_size = settings.EMBEDDING_BATCH_SIZE
            total_tables = len(tables)
            total_upserted = 0

            logger.info(f"Upserting {total_tables} tables...")

            for batch_idx in range(0, total_tables, batch_size):
                batch_tables = tables[batch_idx:batch_idx + batch_size]
                batch_count = len(batch_tables)

                # Generate embeddings for this batch
                texts = []
                for table in batch_tables:
                    table_name = table.get('table_name', '')

                    # Create semantic text for embedding (table name + description only)
                    # Columns are stored separately in the 'schema' field
                    text_parts = [f"Table: {table_name}"]

                    # Use user-provided description if available, otherwise generate one
                    user_description = table.get('description')
                    if user_description:
                        # Use user-provided description from MSSQL extended properties
                        text_parts.append(f"Description: {user_description}")
                        logger.debug(f"[DESC] {table_name}: Using user description: {user_description[:100]}")
                    else:
                        # Fall back to auto-generated description from table name
                        generated_desc = self._generate_semantic_description(table_name)
                        text_parts.append(f"Description: {generated_desc}")
                        logger.debug(f"[DESC] {table_name}: Using auto-generated description")

                    text = "\n".join(text_parts)
                    texts.append(text)

                # Encode batch
                embeddings = self.model.encode(texts, show_progress_bar=False)

                # Generate deterministic IDs
                batch_ids = [
                    self._generate_id(database_id, t.get('table_name', ''))
                    for t in batch_tables
                ]

                # Prepare batch data with needs_sync=False
                import json
                batch_data = [
                    batch_ids,
                    [database_id] * batch_count,
                    [t.get('table_name', '') for t in batch_tables],
                    [text[:4096] for text in texts],
                    [json.dumps(t.get('columns', []))[:8192] for t in batch_tables],  # Schema as JSON
                    [False] * batch_count,  # needs_sync - set to False after sync
                    [False] * batch_count,  # skipped
                    embeddings.tolist()
                ]

                # Upsert batch (updates existing or inserts new)
                self.collection.upsert(batch_data)
                total_upserted += batch_count
                logger.info(f"Upserted {batch_count} embeddings (total: {total_upserted}/{total_tables})")

            # Flush to ensure updates are persisted
            self.collection.flush()
            logger.info(f"Upsert complete: {total_upserted} tables")

            return self.collection_name, total_upserted

        except Exception as e:
            logger.error(f"Error upserting embeddings: {e}")
            raise

    def get_tables_needing_sync(self, database_id: str) -> List[Dict[str, Any]]:
        """
        Get all tables that need syncing (needs_sync=True) for a specific database.

        Args:
            database_id: MongoDB database ID

        Returns:
            List of tables with needs_sync=True
        """
        if not self.connected or not self.loaded:
            raise ConnectionError("Milvus not connected or collection not loaded")

        try:
            # Query for tables with needs_sync=True for this database
            expr = f'database_id == "{database_id}" && needs_sync == true'

            results = self.collection.query(
                expr=expr,
                output_fields=["database_id", "table_name", "text", "needs_sync"]
            )

            logger.info(f"Found {len(results)} tables needing sync for database {database_id}")
            return results

        except Exception as e:
            logger.error(f"Error querying tables needing sync: {e}")
            raise

    def load_index(self) -> bool:
        """
        Load existing collection (for backward compatibility).

        Returns:
            True if collection exists and loaded
        """
        if not self.connected:
            return False

        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                self.loaded = True
                logger.info(f"Loaded existing collection: {self.collection_name}")
                return True
            else:
                logger.warning(f"Collection {self.collection_name} does not exist")
                return False

        except Exception as e:
            logger.error(f"Error loading collection: {e}")
            return False


# Global singleton instance
_vector_store = None


def get_vector_store() -> MilvusVectorStore:
    """Get Milvus vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = MilvusVectorStore()
        _vector_store.connect()
    return _vector_store
