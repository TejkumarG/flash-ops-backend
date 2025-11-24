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
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="database_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
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

                    # Create rich semantic text for better embedding quality
                    # Start with table name and semantic variations
                    text_parts = [f"Table: {table_name}"]

                    # Add semantic description based on table name
                    text_parts.append(self._generate_semantic_description(table_name))

                    # Add column information with types
                    if 'columns' in table and table['columns']:
                        # List column names
                        col_names = [c.get('name', '') for c in table['columns']]
                        text_parts.append(f"Columns: {', '.join(col_names)}")

                        # Add column types for better understanding
                        col_types = []
                        for col in table['columns']:
                            col_name = col.get('name', '')
                            col_type = col.get('type', '')
                            if col_name and col_type:
                                col_types.append(f"{col_name} ({col_type})")
                        if col_types:
                            text_parts.append(f"Schema: {', '.join(col_types[:10])}")  # Limit to 10 to avoid token limit

                    text = "\n".join(text_parts)
                    texts.append(text)

                # Log sample of embedding texts for debugging
                if batch_num == 1 and len(texts) > 0:
                    logger.info(f"[EMBEDDING TEXT SAMPLE] First table embedding text:\n{texts[0][:500]}")

                # Encode batch
                embeddings = self.model.encode(texts, show_progress_bar=False)
                logger.info(f"Batch {batch_num}/{total_batches}: Generated embeddings with shape {embeddings.shape}")

                # Prepare batch data - simplified schema
                batch_data = [
                    [database_id] * batch_count,  # database_id for all tables in batch
                    [t.get('table_name', '') for t in batch_tables],
                    [text[:4096] for text in texts],  # Store the text used for embedding
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
        Search for similar tables using query, filtered by database_id.

        Args:
            query: Natural language query
            database_id: MongoDB database ID to filter results
            k: Number of results to return (default: settings.VECTOR_SEARCH_TOP_K)
            metric_type: Distance metric (L2 or IP)

        Returns:
            List of similar tables with scores
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

            # Perform search with database_id filter
            results = self.collection.search(
                data=query_embedding.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=k,
                expr=f'database_id == "{database_id}"',  # Filter by database_id
                output_fields=["database_id", "table_name", "text"]
            )

            # Format results
            tables = []
            for hits in results:
                for hit in hits:
                    # Convert L2 distance to score (inverse distance)
                    distance = hit.distance
                    score = 1.0 / (1.0 + distance) if metric_type == "L2" else hit.distance

                    table = {
                        "database_id": hit.entity.get("database_id"),
                        "table_name": hit.entity.get("table_name"),
                        "text": hit.entity.get("text"),
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
