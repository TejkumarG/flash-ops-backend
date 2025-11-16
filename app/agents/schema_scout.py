"""
Schema Scout: Vector search to find top-K relevant tables.
Stage 1 of the pipeline.
"""
from typing import List, Dict, Any

from app.services import get_vector_store
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("schema_scout")


class SchemaScout:
    """Agent for semantic table search using vector embeddings."""

    def __init__(self):
        """Initialize Schema Scout."""
        self.vector_store = get_vector_store()

    def search_tables(
        self,
        query: str,
        database_id: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K most relevant tables using vector similarity.

        Args:
            query: Natural language query
            database_id: MongoDB database ID to filter results
            top_k: Number of tables to return (default: from settings)

        Returns:
            List of table metadata with scores:
            [
                {
                    "database_id": str,
                    "table_name": str,
                    "text": str,
                    "score": float,
                    "distance": float
                },
                ...
            ]
        """
        if top_k is None:
            top_k = settings.VECTOR_SEARCH_TOP_K

        logger.info(f"Searching for top {top_k} tables for database {database_id}, query: {query[:100]}...")

        # Use Milvus vector store to search with database_id filter
        results = self.vector_store.search(query, database_id, k=top_k)

        logger.info(
            f"Schema Scout found {len(results)} tables. "
            f"Top score: {results[0]['score']:.3f}" if results else "No results"
        )

        return results


def create_schema_scout() -> SchemaScout:
    """Factory function to create Schema Scout instance."""
    return SchemaScout()
