"""
Table Selector: Select best 1/2/3 tables from cluster representatives.
Stage 3 of the pipeline.
"""
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations
import re

from app.config import settings
from app.services import get_mongo_client
from app.services.schema_extractor import SchemaExtractor
from app.utils.logger import setup_logger
from app.utils.encryption import decrypt_password

logger = setup_logger("table_selector")


class TableSelector:
    """Agent for selecting optimal table combination (1-3 tables)."""

    def __init__(self, fk_metadata: Optional[Dict] = None):
        """
        Initialize Table Selector.

        Args:
            fk_metadata: Foreign key relationships (will be loaded from metadata)
        """
        self.fk_metadata = fk_metadata or {}
        self.mongo_client = get_mongo_client()
        self._table_metadata_cache = {}  # Cache for table metadata
        self._schema_extractor = None  # Will be initialized when needed

    def select_tables_with_cross_cluster_strategy(
        self,
        clusters: List[List[Dict[str, Any]]],
        database_id: str
    ) -> Dict[str, Any]:
        """
        NEW: Select tables using cross-cluster combination strategy with dynamic scoring.

        This is the improved table selection strategy that:
        1. Generates cross-cluster combinations (NEVER within same cluster)
        2. Applies dynamic threshold-based scoring
        3. Returns top 3 combinations for LLM to choose from

        Args:
            clusters: List of semantic domain clusters from table_clustering
            database_id: Database ID for fetching table metadata and join detection

        Returns:
            Selection result with top 3 combinations:
            {
                "selected_tables": [table1, ...],  # Primary selection
                "tier": int,
                "joins": [...],
                "join_pattern": str,
                "confidence": float,
                "reasoning": str,
                "alternatives": [...]  # Top 2 additional combinations
            }
        """
        logger.info(f"[CROSS-CLUSTER STRATEGY] Starting with {len(clusters)} semantic clusters")

        # Store database_id for use in helper methods
        self._current_database_id = database_id

        # Step 1: Generate all cross-cluster combinations
        all_combinations = self._generate_cross_cluster_combinations(
            clusters,
            database_id,
            max_combinations_per_type=20
        )

        if not all_combinations:
            logger.error("[CROSS-CLUSTER STRATEGY] No valid combinations generated!")
            # Fallback: use best table from first cluster
            if clusters and len(clusters[0]) > 0:
                fallback_table = self._enrich_table_with_metadata(clusters[0][0])
                return {
                    "selected_tables": [fallback_table],
                    "tier": 1,
                    "joins": [],
                    "join_pattern": "single",
                    "confidence": clusters[0][0].get("score", 0),
                    "reasoning": "Fallback to highest-scoring table (no valid combinations)",
                    "alternatives": []
                }

        # Step 2: Select top 3 combinations using dynamic threshold scoring
        top_3_combinations = self._select_top_combinations_with_dynamic_scoring(
            all_combinations,
            top_n=3
        )

        if not top_3_combinations:
            logger.error("[CROSS-CLUSTER STRATEGY] No combinations selected!")
            return self.select_tables(clusters[0][:10] if clusters else [], database_id)  # Fallback

        # Primary selection
        primary = top_3_combinations[0]
        logger.info(
            f"[PRIMARY SELECTION] {[t.get('table_name') for t in primary['selected_tables']]} "
            f"(Tier {primary['tier']}, Score: {primary['confidence']:.3f}, Pattern: {primary.get('cluster_pattern', 'unknown')})"
        )

        # Alternative selections
        alternatives = top_3_combinations[1:3] if len(top_3_combinations) > 1 else []
        logger.info(f"[ALTERNATIVES] {len(alternatives)} alternative combinations selected")

        for idx, alt in enumerate(alternatives, 2):
            logger.info(
                f"  Alt #{idx}: {[t.get('table_name') for t in alt['selected_tables']]} "
                f"(Tier {alt['tier']}, Score: {alt['confidence']:.3f})"
            )

        primary["alternatives"] = alternatives
        return primary

    def _enrich_table_with_metadata(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich table dictionary with column metadata from MSSQL.

        Args:
            table: Table dict with at least {table_name, score}

        Returns:
            Enriched table dict with columns info added
        """
        table_name = table.get("table_name")
        if not table_name or not self._current_database_id:
            logger.warning(f"Cannot enrich table {table_name}, missing name or database_id")
            return table

        # Get metadata from MSSQL
        metadata = self._get_table_metadata(table_name, self._current_database_id)

        if metadata and 'columns' in metadata:
            # Add columns to table dict
            enriched = table.copy()
            enriched['columns'] = metadata['columns']
            if 'row_count' in metadata:
                enriched['row_count'] = metadata['row_count']
            logger.info(f"[TABLE ENRICHED] {table_name}: {len(metadata['columns'])} columns, {metadata.get('row_count', 0)} rows")
            logger.debug(f"[ENRICHED COLUMNS] {table_name}: {[col['name'] for col in metadata['columns']]}")
            return enriched
        else:
            logger.warning(f"[TABLE ENRICHMENT FAILED] No metadata found for table {table_name}")
            return table

    def _get_table_metadata(self, table_name: str, database_id: str) -> Optional[Dict[str, Any]]:
        """
        Get table metadata from MSSQL with caching.

        Args:
            table_name: Table name
            database_id: Database ID

        Returns:
            Table metadata with columns info
        """
        cache_key = f"{database_id}:{table_name}"

        if cache_key in self._table_metadata_cache:
            return self._table_metadata_cache[cache_key]

        try:
            # Get connection config
            config_data = self.mongo_client.get_database_connection_config(database_id)
            database_name = config_data["database_name"]
            connection_config = config_data["connection_config"]

            # Initialize schema extractor if needed
            if self._schema_extractor is None:
                self._schema_extractor = SchemaExtractor(connection_config)

            # Fetch schema from MSSQL
            columns = self._schema_extractor.get_table_schema(table_name, database_name)
            row_count = self._schema_extractor.get_row_count(table_name, database_name)

            logger.info(f"[SCHEMA FETCH] Table: {table_name}, Columns: {len(columns)}, Row count: {row_count}")
            logger.debug(f"[SCHEMA DETAILS] {table_name} columns: {[col['name'] for col in columns]}")

            metadata = {
                'columns': columns,
                'row_count': row_count
            }

            self._table_metadata_cache[cache_key] = metadata
            return metadata
        except Exception as e:
            logger.warning(f"Could not fetch metadata for {table_name}: {e}")
            return None

    def _find_join(
        self,
        table1: str,
        table2: str,
        database_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find join condition between two tables.

        Priority:
        1. Foreign key (from metadata) - Confidence: 100%
        2. Column name matching - Confidence: 80%
        3. Column pattern matching - Confidence: 60%

        Args:
            table1: First table name
            table2: Second table name
            database_id: Database ID for fetching metadata

        Returns:
            Join info dict or None
        """
        if table1 == table2:
            return None

        # Get metadata for both tables
        meta1 = self._get_table_metadata(table1, database_id) if database_id else None
        meta2 = self._get_table_metadata(table2, database_id) if database_id else None

        if not meta1 or not meta2:
            logger.debug(f"Metadata not available for {table1} or {table2}, using simple join")
            # Fallback: simple pattern-based join
            return self._simple_join_pattern(table1, table2)

        # Priority 1: Check for FK relationships
        fk_join = self._check_foreign_keys(table1, table2, meta1, meta2)
        if fk_join:
            return fk_join

        # Priority 2: Check for exact column name matches
        column_join = self._check_column_matches(table1, table2, meta1, meta2)
        if column_join:
            return column_join

        # Priority 3: Check for pattern matches
        pattern_join = self._check_pattern_matches(table1, table2, meta1, meta2)
        if pattern_join:
            return pattern_join

        logger.debug(f"No join found between {table1} and {table2}")
        return None

    def _check_foreign_keys(
        self,
        table1: str,
        table2: str,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check for FK relationships between tables."""
        columns1 = meta1.get('columns', [])
        columns2 = meta2.get('columns', [])

        # Check if table1 has FK to table2
        for col in columns1:
            if col.get('is_foreign_key') and col.get('references'):
                ref = col['references']  # Format: "table.column"
                if ref.startswith(f"{table2}."):
                    ref_col = ref.split('.')[1]
                    return {
                        "from_table": table1,
                        "to_table": table2,
                        "condition": f"{table1}.{col['name']} = {table2}.{ref_col}",
                        "type": "foreign_key",
                        "confidence": 1.0
                    }

        # Check if table2 has FK to table1
        for col in columns2:
            if col.get('is_foreign_key') and col.get('references'):
                ref = col['references']
                if ref.startswith(f"{table1}."):
                    ref_col = ref.split('.')[1]
                    return {
                        "from_table": table2,
                        "to_table": table1,
                        "condition": f"{table2}.{col['name']} = {table1}.{ref_col}",
                        "type": "foreign_key",
                        "confidence": 1.0
                    }

        return None

    def _check_column_matches(
        self,
        table1: str,
        table2: str,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check for exact column name matches."""
        columns1 = {col['name']: col for col in meta1.get('columns', [])}
        columns2 = {col['name']: col for col in meta2.get('columns', [])}

        # Find common column names (likely join keys)
        common = set(columns1.keys()) & set(columns2.keys())

        # Filter for likely join keys (id, _id, etc.)
        join_candidates = [c for c in common if 'id' in c.lower()]

        if join_candidates:
            join_col = join_candidates[0]  # Use first match
            return {
                "from_table": table1,
                "to_table": table2,
                "condition": f"{table1}.{join_col} = {table2}.{join_col}",
                "type": "column_match",
                "confidence": 0.8
            }

        return None

    def _check_pattern_matches(
        self,
        table1: str,
        table2: str,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check for pattern-based column matches (e.g., emp_id matches employee.id)."""
        columns1 = meta1.get('columns', [])
        columns2 = meta2.get('columns', [])

        # Common patterns: table_id, table_name_id, tableid
        patterns = [
            (f"{table2}_id", "id"),  # sales.employee_id -> employee.id
            (f"{table2}id", "id"),   # sales.employeeid -> employee.id
            ("id", f"{table1}_id"),  # employee.id -> sales.employee_id
            ("id", f"{table1}id"),   # employee.id -> sales.employeeid
        ]

        for col1 in columns1:
            for col2 in columns2:
                col1_name = col1['name'].lower()
                col2_name = col2['name'].lower()

                # Check if names match patterns
                for pattern1, pattern2 in patterns:
                    if col1_name == pattern1.lower() and col2_name == pattern2.lower():
                        return {
                            "from_table": table1,
                            "to_table": table2,
                            "condition": f"{table1}.{col1['name']} = {table2}.{col2['name']}",
                            "type": "pattern_match",
                            "confidence": 0.6
                        }

        return None

    def _simple_join_pattern(self, table1: str, table2: str) -> Optional[Dict[str, Any]]:
        """Fallback: Simple pattern-based join when metadata is unavailable."""
        # Common pattern: table2 has table1_id column
        return {
            "from_table": table1,
            "to_table": table2,
            "condition": f"{table1}.id = {table2}.{table1}_id",
            "type": "inferred",
            "confidence": 0.5
        }

    def _score_combination(
        self,
        tables: List[Dict[str, Any]],
        joins: List[Dict[str, Any]]
    ) -> float:
        """
        Score a table combination.

        Args:
            tables: List of selected tables
            joins: List of join conditions

        Returns:
            Combined score (0-1)
        """
        # Component 1: Average semantic score (60%)
        avg_score = sum(t.get("score", 0) for t in tables) / len(tables)
        semantic_component = avg_score * 0.6

        # Component 2: Join quality (30%)
        join_quality = 0
        for join in joins:
            if join.get("type") == "foreign_key":
                join_quality += 1.2
            elif join.get("type") == "column_match":
                join_quality += 1.0
            else:
                join_quality += 0.8
        join_quality = (join_quality / len(joins)) if joins else 1.0
        join_component = (join_quality / 1.2) * 0.3  # Normalize

        # Component 3: Pattern bonus (10%)
        pattern_component = 0.1

        total_score = semantic_component + join_component + pattern_component
        return min(total_score, 1.0)

    def _get_dynamic_threshold(self, base_score: float) -> float:
        """
        Calculate dynamic threshold based on confidence level.

        Score-adaptive thresholds:
        - High scores (0.7+): Larger threshold (prefer simplicity)
        - Medium scores (0.5-0.7): Medium threshold (balanced)
        - Medium-low scores (0.3-0.5): Smaller threshold (favor context)
        - Low scores (< 0.3): Very small threshold (need more context)

        Args:
            base_score: The base score to evaluate

        Returns:
            Dynamic threshold value
        """
        if base_score >= 0.7:
            # High confidence - prefer simplicity
            return 0.08
        elif base_score >= 0.5:
            # Medium-high confidence - balanced
            return 0.06
        elif base_score >= 0.3:
            # Medium-low confidence - favor complexity
            return 0.04
        else:
            # Low confidence - strongly favor more context
            return 0.03

    def _generate_cross_cluster_combinations(
        self,
        clusters: List[List[Dict[str, Any]]],
        database_id: str,
        max_combinations_per_type: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generate cross-cluster combinations (NEVER within same cluster).

        Generates:
        - Single table combos: One from each cluster
        - 2-table combos: One from cluster 1 + one from cluster 2 (cross-cluster only)
        - 3-table combos: One from each of 3 clusters (cross-cluster only)

        Args:
            clusters: List of semantic domain clusters
            database_id: Database ID for join detection
            max_combinations_per_type: Max combinations to generate per type

        Returns:
            List of all valid cross-cluster combinations
        """
        all_combinations = []

        logger.info(f"[CROSS-CLUSTER] Generating combinations from {len(clusters)} clusters")
        for idx, cluster in enumerate(clusters):
            logger.info(f"  Cluster {idx+1}: {len(cluster)} tables")

        # Strategy 1: Single table combinations (top tables from each cluster)
        logger.info(f"[STRATEGY 1] Generating single table combinations")
        for cluster_idx, cluster in enumerate(clusters):
            # Take top 5 from each cluster
            for table in cluster[:min(5, len(cluster))]:
                enriched_table = self._enrich_table_with_metadata(table)
                all_combinations.append({
                    "selected_tables": [enriched_table],
                    "tier": 1,
                    "joins": [],
                    "join_pattern": "single",
                    "confidence": table.get("score", 0),
                    "reasoning": f"Single table from cluster {cluster_idx+1} (score: {table.get('score', 0):.3f})",
                    "cluster_pattern": f"C{cluster_idx+1}"
                })

        logger.info(f"  Generated {len(all_combinations)} single-table combinations")

        # Strategy 2: 2-table cross-cluster combinations
        if len(clusters) >= 2:
            logger.info(f"[STRATEGY 2] Generating 2-table cross-cluster combinations")
            two_table_count = 0

            # Only combine tables from DIFFERENT clusters
            for i in range(len(clusters) - 1):
                for j in range(i + 1, len(clusters)):
                    # Limit combinations per cluster pair
                    pair_count = 0

                    for table1 in clusters[i][:min(3, len(clusters[i]))]:  # Top 3 from cluster i
                        for table2 in clusters[j][:min(3, len(clusters[j]))]:  # Top 3 from cluster j
                            if pair_count >= max_combinations_per_type:
                                break

                            # Check if joinable
                            join_info = self._find_join(
                                table1["table_name"],
                                table2["table_name"],
                                database_id
                            )

                            if join_info:
                                # Calculate score
                                score = self._score_combination([table1, table2], [join_info])

                                # Enrich tables
                                enriched_tables = [
                                    self._enrich_table_with_metadata(table1),
                                    self._enrich_table_with_metadata(table2)
                                ]

                                all_combinations.append({
                                    "selected_tables": enriched_tables,
                                    "tier": 2,
                                    "joins": [join_info],
                                    "join_pattern": "pair",
                                    "confidence": score,
                                    "reasoning": f"Cross-cluster join: C{i+1} + C{j+1}",
                                    "cluster_pattern": f"C{i+1}+C{j+1}"
                                })

                                pair_count += 1
                                two_table_count += 1

            logger.info(f"  Generated {two_table_count} two-table cross-cluster combinations")

        # Strategy 3: 3-table cross-cluster combinations
        if len(clusters) >= 3:
            logger.info(f"[STRATEGY 3] Generating 3-table cross-cluster combinations")
            three_table_count = 0

            # Try combining one from each of first 3 clusters
            for table1 in clusters[0][:min(2, len(clusters[0]))]:
                for table2 in clusters[1][:min(2, len(clusters[1]))]:
                    for table3 in clusters[2][:min(2, len(clusters[2]))]:
                        if three_table_count >= max_combinations_per_type:
                            break

                        # Find joins
                        join1 = self._find_join(table1["table_name"], table2["table_name"], database_id)
                        join2 = self._find_join(table1["table_name"], table3["table_name"], database_id)

                        if join1 and join2:
                            # Star pattern with table1 as center
                            score = self._score_combination([table1, table2, table3], [join1, join2])
                            score += 0.05  # Small bonus for 3-table diversity

                            enriched_tables = [
                                self._enrich_table_with_metadata(table1),
                                self._enrich_table_with_metadata(table2),
                                self._enrich_table_with_metadata(table3)
                            ]

                            all_combinations.append({
                                "selected_tables": enriched_tables,
                                "tier": 3,
                                "joins": [join1, join2],
                                "join_pattern": "star",
                                "confidence": score,
                                "reasoning": f"Cross-cluster star: C1 + C2 + C3 (center: {table1['table_name']})",
                                "cluster_pattern": "C1+C2+C3"
                            })

                            three_table_count += 1

            logger.info(f"  Generated {three_table_count} three-table cross-cluster combinations")

        logger.info(f"[CROSS-CLUSTER TOTAL] Generated {len(all_combinations)} total combinations")

        return all_combinations

    def _select_top_combinations_with_dynamic_scoring(
        self,
        all_combinations: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Select top N combinations using dynamic threshold-based scoring.

        Logic:
        - Sort all combinations by score
        - Apply dynamic threshold to decide between simple vs complex combos
        - Prefer simpler combos unless complex ones are within dynamic threshold

        Args:
            all_combinations: List of all generated combinations
            top_n: Number of top combinations to return

        Returns:
            Top N combinations based on dynamic scoring
        """
        if not all_combinations:
            logger.warning("[DYNAMIC SCORING] No combinations available")
            return []

        # Sort by confidence (descending)
        sorted_combos = sorted(all_combinations, key=lambda x: x["confidence"], reverse=True)

        # Get best score and calculate dynamic threshold
        best_score = sorted_combos[0]["confidence"]
        threshold = self._get_dynamic_threshold(best_score)

        logger.info(f"[DYNAMIC SCORING] Best score: {best_score:.3f}, Dynamic threshold: {threshold:.3f}")
        logger.info(f"[DYNAMIC SCORING] Evaluating {len(sorted_combos)} combinations")

        # Log top 10 for analysis
        for idx, combo in enumerate(sorted_combos[:10]):
            tier = combo["tier"]
            score = combo["confidence"]
            gap = best_score - score
            tables_str = ', '.join([t.get('table_name', 'unknown') for t in combo['selected_tables']])
            cluster_pattern = combo.get('cluster_pattern', 'unknown')

            logger.info(
                f"  #{idx+1}: {tables_str} | "
                f"Tier={tier}, Score={score:.3f}, Gap={gap:.3f}, Pattern={cluster_pattern}"
            )

        # Select top N combinations
        # Prioritize diversity: try to get mix of single/2-table/3-table if scores are close
        selected = []
        tier_counts = {1: 0, 2: 0, 3: 0}

        for combo in sorted_combos:
            if len(selected) >= top_n:
                break

            tier = combo["tier"]
            score = combo["confidence"]
            gap = best_score - score

            # Accept if within dynamic threshold OR if we need diversity
            if gap <= threshold or len(selected) < top_n:
                # Prefer diversity (don't take too many of same tier)
                if tier_counts[tier] < 2 or len(selected) < top_n:  # Max 2 per tier unless we're short
                    selected.append(combo)
                    tier_counts[tier] += 1
                    logger.info(
                        f"[SELECTED #{len(selected)}] Tier {tier}, Score={score:.3f}, "
                        f"Gap={gap:.3f}, Reason={'within_threshold' if gap <= threshold else 'diversity'}"
                    )

        logger.info(
            f"[DYNAMIC SCORING RESULT] Selected {len(selected)} combinations: "
            f"Tier1={tier_counts[1]}, Tier2={tier_counts[2]}, Tier3={tier_counts[3]}"
        )

        return selected


def create_table_selector() -> TableSelector:
    """Factory function to create Table Selector instance."""
    return TableSelector()
