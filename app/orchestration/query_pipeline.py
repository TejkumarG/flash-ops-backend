"""
Query Pipeline Orchestrator: Coordinates all agents for query processing.
PyTorch-style handler pattern.
"""
import time
from typing import Dict, Any

from app.agents import (
    create_schema_scout,
    create_table_clustering,
    create_table_selector,
    create_schema_packager,
    create_sql_generator,
    create_quality_inspector
)
from app.models import QueryResponse, QueryStatus, JoinInfo
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("query_pipeline")


class QueryPipeline:
    """
    Main orchestrator for query processing pipeline.
    Coordinates all agents in sequence (PyTorch-style handler).
    """

    def __init__(self):
        """Initialize pipeline with all agents."""
        logger.info("Initializing Query Pipeline...")

        # Initialize all agents
        self.schema_scout = create_schema_scout()
        self.table_clustering = create_table_clustering()
        self.table_selector = create_table_selector()
        self.schema_packager = create_schema_packager()
        self.sql_generator = create_sql_generator()
        self.quality_inspector = create_quality_inspector()

        logger.info("Query Pipeline initialized successfully")

    def process(self, query: str, database_id: str) -> QueryResponse:
        """
        Process natural language query through the pipeline.

        Args:
            query: Natural language query string
            database_id: MongoDB database ID to query against

        Returns:
            QueryResponse with results or error
        """
        start_time = time.time()
        logger.info(f"Processing query: {query} for database: {database_id}")

        try:
            # Stage 1: Vector Search (Schema Scout)
            logger.info("Stage 1: Vector Search")
            top_tables = self.schema_scout.search_tables(query, database_id)

            if not top_tables:
                return self._error_response(
                    query,
                    "No relevant tables found",
                    ["Try rephrasing your query", "Check if tables are indexed"],
                    int((time.time() - start_time) * 1000)
                )

            # Stage 2: Table Clustering (Semantic Difference Strategy)
            logger.info("Stage 2: Semantic Difference Clustering")
            clusters = self.table_clustering.cluster_tables_by_semantic_difference(top_tables, max_clusters=3)
            logger.info(f"Created {len(clusters)} semantic domain clusters with sizes: {[len(c) for c in clusters]}")

            # Stage 3: Cross-Cluster Table Selection (with Dynamic Threshold Scoring)
            logger.info("Stage 3: Cross-Cluster Table Selection with Dynamic Scoring")
            selection = self.table_selector.select_tables_with_cross_cluster_strategy(clusters, database_id)

            # Collect all combinations (primary + alternatives)
            combinations = [selection]  # Primary selection
            if "alternatives" in selection and selection["alternatives"]:
                combinations.extend(selection["alternatives"])

            logger.info(f"[AVAILABLE COMBINATIONS] {len(combinations)} table combinations available")
            for idx, combo in enumerate(combinations, 1):
                combo_tables = ', '.join([t.get('table_name', 'unknown') for t in combo['selected_tables']])
                logger.info(f"  Option {idx}: {combo_tables} (confidence: {combo['confidence']:.3f}, tier: {combo['tier']})")

            # Retry logic: max 2 attempts, exclude used combinations
            MAX_ATTEMPTS = 2
            used_combination_indices = []
            available_combinations = combinations.copy()

            for attempt in range(1, MAX_ATTEMPTS + 1):
                logger.info(f"[ATTEMPT {attempt}/{MAX_ATTEMPTS}] Trying with {len(available_combinations)} combinations")

                # Stage 4: Schema Packaging - Pass available combinations to LLM
                schema_package = self.schema_packager.package_multiple_schemas(available_combinations, query)

                # Stage 5: SQL Generation - LLM chooses best combination and generates SQL
                sql, errors = self._generate_sql_with_reflection(schema_package, database_id)

                if not sql:
                    logger.warning(f"[ATTEMPT {attempt}] SQL generation failed: {errors}")
                    if attempt == MAX_ATTEMPTS:
                        return self._error_response(
                            query,
                            f"SQL generation failed: {errors[0] if errors else 'Unknown error'}",
                            ["Check table schemas", "Try rephrasing your query"],
                            int((time.time() - start_time) * 1000)
                        )
                    continue

                # Stage 6-7: Validation and Execution
                is_valid, error_msg, results, row_count = self.quality_inspector.validate_and_execute(sql, database_id)

                if not is_valid:
                    logger.warning(f"[ATTEMPT {attempt}] SQL validation/execution failed: {error_msg}")
                    if attempt == MAX_ATTEMPTS:
                        return self._error_response(
                            query,
                            f"SQL validation/execution failed: {error_msg}",
                            ["Check SQL syntax", "Verify table/column names"],
                            int((time.time() - start_time) * 1000)
                        )
                    continue

                # Identify which combination was used by matching tables in SQL
                used_combo, used_combo_idx = self._identify_used_combination(sql, available_combinations)

                if used_combo:
                    combo_tables = [t.get('table_name', 'unknown') for t in used_combo['selected_tables']]
                    logger.info(f"[ATTEMPT {attempt}] LLM selected combination: {', '.join(combo_tables)}")
                else:
                    # Fallback: use first combination
                    combo_tables = [t.get('table_name', 'unknown') for t in available_combinations[0]['selected_tables']]
                    used_combo = available_combinations[0]

                # Check for 0 results
                if row_count == 0 and attempt < MAX_ATTEMPTS:
                    logger.warning(f"[ATTEMPT {attempt}] Query returned 0 rows, will retry with different combinations")

                    # Remove the used combination from available options
                    if used_combo_idx is not None:
                        original_idx = combinations.index(available_combinations[used_combo_idx])
                        used_combination_indices.append(original_idx)
                        available_combinations = [c for i, c in enumerate(combinations) if i not in used_combination_indices]

                        if not available_combinations:
                            logger.warning(f"[ATTEMPT {attempt}] No more combinations to try")
                            break

                        logger.info(f"[RETRY] Excluding combination {original_idx + 1}, {len(available_combinations)} combinations remaining")
                    continue

                # Success or last attempt - return result
                if row_count == 0:
                    logger.warning(f"[FINAL ATTEMPT] Attempt {attempt} returned 0 rows, showing result anyway")
                    json_results = []
                    file_path = None
                    formatted_result = "No results found for this query."
                else:
                    json_results, file_path, formatted_result = self.quality_inspector.format_response(
                        sql, results, row_count, query, combo_tables, database_id
                    )

                execution_time = int((time.time() - start_time) * 1000)

                response = QueryResponse(
                    status=QueryStatus.SUCCESS,
                    query=query,
                    tables_used=combo_tables,
                    tier=used_combo.get("tier", 1),
                    row_count=row_count,
                    result=json_results,
                    csv_path=None,
                    sql_generated=sql,
                    joins=self._format_joins(used_combo.get("joins", [])),
                    execution_time_ms=execution_time,
                    confidence=used_combo.get("confidence", 0.5),
                    formatted_result=formatted_result,
                    file_path=file_path
                )

                logger.info(f"[SUCCESS] Attempt {attempt} completed: {row_count} rows in {execution_time}ms")
                return response

            # All attempts failed
            return self._error_response(
                query,
                f"Failed after {MAX_ATTEMPTS} attempts with different table combinations",
                ["Try rephrasing your query", "Check if data exists for this query"],
                int((time.time() - start_time) * 1000)
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._error_response(
                query,
                f"Internal error: {str(e)}",
                ["Contact support if issue persists"],
                int((time.time() - start_time) * 1000)
            )

    def _generate_sql_with_reflection(
        self,
        schema_package: str,
        database_id: str
    ) -> tuple[str, list]:
        """
        Generate SQL with reflection (up to 3 attempts).

        Args:
            schema_package: Formatted schema package
            database_id: MongoDB database ID

        Returns:
            Tuple of (sql, errors)
        """
        from app.services import get_mongo_client
        from app.services.schema_extractor import SchemaExtractor

        errors = []

        # Get connection config for validation
        mongo_client = get_mongo_client()
        config_data = mongo_client.get_database_connection_config(database_id)
        database_name = config_data["database_name"]
        connection_config = config_data["connection_config"]

        # Create schema extractor for validation
        extractor = SchemaExtractor(connection_config)

        try:
            for attempt in range(1, settings.MAX_REFLECTION_ATTEMPTS + 1):
                try:
                    sql = self.sql_generator.generate_sql(
                        schema_package,
                        attempt=attempt,
                        previous_errors=errors
                    )

                    # Quick validation using actual MSSQL connection
                    is_valid, error_msg = extractor.validate_sql(sql, database_name)

                    if is_valid:
                        extractor.close()
                        return sql, []
                    else:
                        logger.warning(f"Attempt {attempt} validation failed: {error_msg}")
                        errors.append(error_msg)

                except Exception as e:
                    logger.error(f"Attempt {attempt} error: {e}")
                    errors.append(str(e))

            extractor.close()
            return None, errors

        except Exception as e:
            extractor.close()
            logger.error(f"Error in SQL reflection: {e}")
            return None, [str(e)]

    def _identify_used_combination(
        self,
        sql: str,
        combinations: list
    ) -> tuple:
        """
        Identify which combination was used by parsing SQL and matching tables.

        Args:
            sql: Generated SQL query
            combinations: List of available combinations

        Returns:
            Tuple of (combination_dict, index) or (None, None) if no match
        """
        import re

        # Extract table names from SQL (simple pattern matching)
        # Match FROM/JOIN tablename or FROM/JOIN tablename alias
        table_pattern = r'(?:FROM|JOIN)\s+([a-zA-Z0-9_\$\'\-]+)'
        sql_tables = re.findall(table_pattern, sql, re.IGNORECASE)

        # Clean table names (remove quotes, aliases)
        sql_tables = [t.strip().strip("'\"") for t in sql_tables]
        sql_tables_set = set(sql_tables)

        logger.debug(f"[COMBO MATCH] SQL uses tables: {sql_tables_set}")

        # Try to match with combinations
        for idx, combo in enumerate(combinations):
            combo_tables = set(t.get('table_name', '') for t in combo['selected_tables'])
            logger.debug(f"[COMBO MATCH] Option {idx + 1} has tables: {combo_tables}")

            # Check if all combo tables are in SQL (allowing for partial matches)
            if combo_tables.issubset(sql_tables_set) or sql_tables_set.issubset(combo_tables):
                logger.info(f"[COMBO MATCH] Matched to combination {idx + 1}")
                return combo, idx

        logger.warning(f"[COMBO MATCH] Could not match SQL tables to any combination")
        return None, None

    def _format_joins(self, joins: list) -> list:
        """Format joins for response."""
        return [
            JoinInfo(
                from_table=j.get("from_table", ""),
                to_table=j.get("to_table", ""),
                condition=j.get("condition", ""),
                type=j.get("type", "unknown")
            )
            for j in joins
        ]

    def _error_response(
        self,
        query: str,
        error_msg: str,
        suggestions: list,
        execution_time: int
    ) -> QueryResponse:
        """Create error response."""
        return QueryResponse(
            status=QueryStatus.ERROR,
            query=query,
            tables_used=[],
            tier=1,
            row_count=0,
            sql_generated="",
            execution_time_ms=execution_time,
            confidence=0.0,
            error_message=error_msg,
            suggestions=suggestions
        )


# Global singleton instance
_query_pipeline = None


def get_query_pipeline() -> QueryPipeline:
    """Get Query Pipeline singleton."""
    global _query_pipeline
    if _query_pipeline is None:
        _query_pipeline = QueryPipeline()
    return _query_pipeline
