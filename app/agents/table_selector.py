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

    def select_tables(
        self,
        cluster_representatives: List[Dict[str, Any]],
        database_id: str = None
    ) -> Dict[str, Any]:
        """
        Select best combination of 1, 2, or 3 tables using range-based scoring.

        Args:
            cluster_representatives: List of representative tables from clustering
            database_id: Database ID for fetching table metadata

        Returns:
            Selection result:
            {
                "selected_tables": [table1, table2, ...],
                "tier": int (1-3),
                "joins": [{"from": ..., "to": ..., "condition": ..., "type": ...}],
                "join_pattern": str ("single", "pair", "star"),
                "confidence": float,
                "reasoning": str
            }
        """
        logger.info(f"Selecting tables from {len(cluster_representatives)} candidates")

        # Store database_id for use in join detection
        self._current_database_id = database_id

        # Get top table scores for comparison
        top_score = cluster_representatives[0].get("score", 0) if cluster_representatives else 0
        second_score = cluster_representatives[1].get("score", 0) if len(cluster_representatives) > 1 else 0
        score_gap = top_score - second_score

        logger.info(f"[TABLE SELECTION] Top table: {cluster_representatives[0].get('table_name')} (score: {top_score:.3f})")
        logger.info(f"[TABLE SELECTION] 2nd table: {cluster_representatives[1].get('table_name') if len(cluster_representatives) > 1 else 'N/A'} (score: {second_score:.3f})")
        logger.info(f"[TABLE SELECTION] Score gap: {score_gap:.3f}")

        # Strategy 1: Very high single-table confidence (≥0.85) - use immediately
        if top_score >= 0.85:
            logger.info(f"[DECISION] Single table selected: Very high confidence {top_score:.3f} ≥ 0.85")
            enriched_table = self._enrich_table_with_metadata(cluster_representatives[0])
            return {
                "selected_tables": [enriched_table],
                "tier": 1,
                "joins": [],
                "join_pattern": "single",
                "confidence": top_score,
                "reasoning": f"Very high single-table confidence ({top_score:.3f})"
            }

        # Strategy 2: Try 2-table combinations and compare with single-table
        two_table_result = self._try_two_tables(cluster_representatives)

        if two_table_result:
            two_table_score = two_table_result["confidence"]
            logger.info(f"[TABLE SELECTION] 2-table score: {two_table_score:.3f}")

            # If 2-table score is very high (≥0.90), use it
            if two_table_score >= 0.90:
                logger.info(f"[DECISION] 2-table selected: Very high 2-table confidence {two_table_score:.3f} ≥ 0.90")
                return two_table_result

            # If single table has huge margin (≥0.2) and decent score (≥0.70), prefer single
            if score_gap >= 0.2 and top_score >= 0.70:
                logger.info(f"[DECISION] Single table selected: Large gap {score_gap:.3f} with decent score {top_score:.3f}")
                enriched_table = self._enrich_table_with_metadata(cluster_representatives[0])
                return {
                    "selected_tables": [enriched_table],
                    "tier": 1,
                    "joins": [],
                    "join_pattern": "single",
                    "confidence": top_score,
                    "reasoning": f"Large score gap ({score_gap:.3f}) with decent confidence ({top_score:.3f})"
                }

            # Otherwise, use 2-table result
            logger.info(f"[DECISION] 2-table selected: 2-table score {two_table_score:.3f} better than single {top_score:.3f}")
            return two_table_result

        # Try 3-table combinations
        three_table_result = self._try_three_tables(cluster_representatives)
        if three_table_result:
            return three_table_result

        # Fallback: Use best single table
        logger.warning("[DECISION] Fallback to single table - no valid multi-table combination")
        enriched_table = self._enrich_table_with_metadata(cluster_representatives[0])
        return {
            "selected_tables": [enriched_table],
            "tier": 1,
            "joins": [],
            "join_pattern": "single",
            "confidence": top_score,
            "reasoning": "Fallback to highest-scoring table"
        }

    def _try_single_table(
        self,
        tables: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if single table is sufficient based on score gap.

        Args:
            tables: List of candidate tables

        Returns:
            Selection result if single table is sufficient, None otherwise
        """
        if len(tables) < 2:
            # Only one table available
            enriched_table = self._enrich_table_with_metadata(tables[0])
            return {
                "selected_tables": [enriched_table],
                "tier": 1,
                "joins": [],
                "join_pattern": "single",
                "confidence": tables[0].get("score", 0.5),
                "reasoning": "Only one table available"
            }

        # Check score gap
        top_score = tables[0].get("score", 0)
        second_score = tables[1].get("score", 0)
        gap = top_score - second_score

        logger.debug(f"Score gap: {gap:.3f} (threshold: {settings.SINGLE_TABLE_SCORE_GAP})")

        if gap > settings.SINGLE_TABLE_SCORE_GAP:
            logger.info(f"Large score gap detected ({gap:.3f}), using single table")
            enriched_table = self._enrich_table_with_metadata(tables[0])
            return {
                "selected_tables": [enriched_table],
                "tier": 1,
                "joins": [],
                "join_pattern": "single",
                "confidence": top_score,
                "reasoning": f"Large score gap ({gap:.3f}) suggests single table sufficient"
            }

        return None

    def _try_two_tables(
        self,
        tables: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Try all 2-table combinations and pick the best.

        Args:
            tables: List of candidate tables

        Returns:
            Best 2-table combination or None
        """
        if len(tables) < 2:
            return None

        logger.info("Trying 2-table combinations...")

        valid_combinations = []

        # Try all pairs
        for table1, table2 in combinations(tables[:10], 2):  # Limit to top 10
            # Check if joinable
            join_info = self._find_join(
                table1["table_name"],
                table2["table_name"],
                self._current_database_id
            )

            if join_info:
                # Calculate score
                score = self._score_combination([table1, table2], [join_info])
                valid_combinations.append({
                    "tables": [table1, table2],
                    "joins": [join_info],
                    "score": score
                })

        if not valid_combinations:
            logger.info("No valid 2-table combinations found")
            return None

        # Pick best combination
        best = max(valid_combinations, key=lambda x: x["score"])

        logger.info(
            f"Best 2-table combination: {[t['table_name'] for t in best['tables']]} "
            f"(score: {best['score']:.3f})"
        )

        # Enrich tables with metadata
        enriched_tables = [self._enrich_table_with_metadata(t) for t in best["tables"]]

        return {
            "selected_tables": enriched_tables,
            "tier": 2,
            "joins": best["joins"],
            "join_pattern": "pair",
            "confidence": best["score"],
            "reasoning": "Best 2-table combination with valid join"
        }

    def _try_three_tables(
        self,
        tables: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Try 3-table combinations (star schema preferred).

        Args:
            tables: List of candidate tables

        Returns:
            Best 3-table combination or None
        """
        if len(tables) < 3:
            return None

        logger.info("Trying 3-table combinations (star schema)...")

        valid_combinations = []

        # Try each table as star center
        for center_idx in range(min(5, len(tables))):  # Top 5 as potential centers
            center = tables[center_idx]

            # Find tables that join to center
            joinable = []
            for other in tables:
                if other == center:
                    continue

                join_info = self._find_join(
                    center["table_name"],
                    other["table_name"],
                    self._current_database_id
                )
                if join_info:
                    joinable.append({"table": other, "join": join_info})

            # Need at least 2 tables to join with center
            if len(joinable) >= 2:
                # Take top 2 by score
                joinable.sort(key=lambda x: x["table"].get("score", 0), reverse=True)
                support1, support2 = joinable[0], joinable[1]

                # Create 3-table star combination
                selected_tables = [center, support1["table"], support2["table"]]
                joins = [support1["join"], support2["join"]]

                score = self._score_combination(selected_tables, joins)
                score += 0.1  # Star schema bonus

                valid_combinations.append({
                    "tables": selected_tables,
                    "joins": joins,
                    "score": score,
                    "center": center["table_name"]
                })

        if not valid_combinations:
            logger.info("No valid 3-table combinations found")
            return None

        # Pick best combination
        best = max(valid_combinations, key=lambda x: x["score"])

        logger.info(
            f"Best 3-table combination: {[t['table_name'] for t in best['tables']]} "
            f"(center: {best['center']}, score: {best['score']:.3f})"
        )

        # Enrich tables with metadata
        enriched_tables = [self._enrich_table_with_metadata(t) for t in best["tables"]]

        return {
            "selected_tables": enriched_tables,
            "tier": 3,
            "joins": best["joins"],
            "join_pattern": "star",
            "confidence": best["score"],
            "reasoning": f"Best 3-table star schema with {best['center']} at center"
        }

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


def create_table_selector() -> TableSelector:
    """Factory function to create Table Selector instance."""
    return TableSelector()
