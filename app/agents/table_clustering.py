"""
Table Clustering: Group tables into semantically different domains.
Stage 2 of the pipeline.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from app.services import get_vector_store
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("table_clustering")


class TableClustering:
    """Agent for clustering tables into semantically different domains."""

    def __init__(self):
        """Initialize Table Clustering agent."""
        self.vector_store = get_vector_store()

    def cluster_tables_by_semantic_difference(
        self,
        tables: List[Dict[str, Any]],
        max_clusters: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster tables into semantically DIFFERENT domains using tiered approach.

        Strategy:
        - Group tables by semantic similarity to create topic clusters
        - Each cluster represents a different semantic domain
        - Returns full clusters (not just representatives)

        Example for 30 tables:
        - Cluster 1: 15 tables (primary domain - highest scores)
        - Cluster 2: 10 tables (secondary domain)
        - Cluster 3: 5 tables (tertiary domain)

        Args:
            tables: List of table metadata from Schema Scout (sorted by score)
            max_clusters: Maximum number of semantic clusters to create (default: 3)

        Returns:
            List of clusters, each containing ALL tables in that semantic domain:
            [
                [table1, table2, ..., table15],  # Cluster 1 (primary domain)
                [table16, table17, ..., table25], # Cluster 2 (secondary domain)
                [table26, table27, ..., table30]  # Cluster 3 (tertiary domain)
            ]
        """
        if len(tables) <= 1:
            logger.info(f"Only {len(tables)} table(s), returning as single cluster")
            return [tables]

        logger.info(f"Clustering {len(tables)} tables into max {max_clusters} semantic domains")

        # Get embeddings for all tables
        embeddings = self._get_table_embeddings(tables)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix

        # Apply agglomerative clustering with fixed number of clusters
        n_clusters = min(max_clusters, len(tables))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )

        labels = clustering.fit_predict(distance_matrix)

        # Group tables by cluster label
        cluster_dict = {}
        for idx, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(tables[idx])

        # Convert to list and sort clusters by size (descending)
        clusters = sorted(cluster_dict.values(), key=len, reverse=True)

        logger.info(f"Created {len(clusters)} semantic domain clusters")
        for i, cluster in enumerate(clusters):
            avg_score = sum(t.get('score', 0) for t in cluster) / len(cluster)
            table_names = [t['table_name'] for t in cluster[:3]]
            logger.info(
                f"Cluster {i+1}: {len(cluster)} tables, avg_score={avg_score:.3f}, "
                f"sample={table_names}{'...' if len(cluster) > 3 else ''}"
            )

        return clusters

    def _get_table_embeddings(self, tables: List[Dict[str, Any]]) -> np.ndarray:
        """
        Get embeddings for tables (re-encode descriptions).

        Args:
            tables: List of table metadata

        Returns:
            Numpy array of embeddings
        """
        # Create descriptions
        descriptions = [
            self._create_description(table)
            for table in tables
        ]

        # Encode
        self.vector_store.load_encoder()
        embeddings = self.vector_store.encoder.encode(
            descriptions,
            convert_to_numpy=True
        )

        return embeddings

    def _create_description(self, table: Dict[str, Any]) -> str:
        """Create description for encoding."""
        parts = [table["table_name"]]
        if table.get("description"):
            parts.append(table["description"])
        if table.get("category"):
            parts.append(f"Category: {table['category']}")
        return ". ".join(parts)


def create_table_clustering() -> TableClustering:
    """Factory function to create Table Clustering instance."""
    return TableClustering()
