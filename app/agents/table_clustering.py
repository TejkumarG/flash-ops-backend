"""
Table Clustering: Group similar tables together.
Stage 2 of the pipeline.
"""
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from app.services import get_vector_store
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("table_clustering")


class TableClustering:
    """Agent for clustering similar tables together."""

    def __init__(self):
        """Initialize Table Clustering agent."""
        self.vector_store = get_vector_store()

    def cluster_tables(
        self,
        tables: List[Dict[str, Any]],
        threshold: float = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster tables by semantic similarity.

        Args:
            tables: List of table metadata from Schema Scout
            threshold: Similarity threshold for clustering (default: from settings)

        Returns:
            List of clusters, each cluster is a list of tables:
            [
                [table1, table2, table3],  # Cluster 1
                [table4, table5],          # Cluster 2
                [table6],                  # Cluster 3 (single table)
                ...
            ]
        """
        if threshold is None:
            threshold = settings.CLUSTERING_SIMILARITY_THRESHOLD

        if len(tables) <= 1:
            return [tables]

        logger.info(f"Clustering {len(tables)} tables with threshold {threshold}")

        # Get embeddings for all tables
        embeddings = self._get_table_embeddings(tables)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix

        # Apply agglomerative clustering
        distance_threshold = 1 - threshold  # Convert similarity to distance
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )

        labels = clustering.fit_predict(distance_matrix)

        # Group tables by cluster label
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tables[idx])

        # Convert to list of clusters
        cluster_list = list(clusters.values())

        logger.info(f"Created {len(cluster_list)} clusters")
        for i, cluster in enumerate(cluster_list):
            logger.debug(
                f"Cluster {i}: {len(cluster)} tables - "
                f"{[t['table_name'] for t in cluster[:3]]}"
            )

        return cluster_list

    def get_cluster_representatives(
        self,
        clusters: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Get best representative from each cluster (highest score).

        Args:
            clusters: List of table clusters

        Returns:
            List of representative tables (one per cluster)
        """
        representatives = []

        for cluster in clusters:
            # Sort by score (descending) and pick the best
            best_table = max(cluster, key=lambda t: t.get('score', 0))
            representatives.append(best_table)

        logger.info(
            f"Selected {len(representatives)} cluster representatives: "
            f"{[t['table_name'] for t in representatives]}"
        )

        return representatives

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
