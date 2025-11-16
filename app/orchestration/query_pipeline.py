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

            # Stage 2: Table Clustering
            logger.info("Stage 2: Table Clustering")
            clusters = self.table_clustering.cluster_tables(top_tables)
            cluster_representatives = self.table_clustering.get_cluster_representatives(clusters)

            # Stage 3: Table Selection
            logger.info("Stage 3: Table Selection")
            selection = self.table_selector.select_tables(cluster_representatives, database_id)

            # Stage 4: Schema Packaging
            logger.info("Stage 4: Schema Packaging")
            schema_package = self.schema_packager.package_schemas(selection, query)

            # Stage 5: SQL Generation (with reflection)
            logger.info("Stage 5: SQL Generation")
            sql, errors = self._generate_sql_with_reflection(schema_package, database_id)

            if not sql:
                return self._error_response(
                    query,
                    f"Failed to generate valid SQL after {settings.MAX_REFLECTION_ATTEMPTS} attempts",
                    [f"Errors: {', '.join(errors)}"],
                    int((time.time() - start_time) * 1000)
                )

            # Stage 6 & 7: Validation and Execution
            logger.info("Stage 6-7: Validation and Execution")
            is_valid, error_msg, results, row_count = self.quality_inspector.validate_and_execute(sql, database_id)

            if not is_valid:
                return self._error_response(
                    query,
                    f"SQL validation/execution failed: {error_msg}",
                    ["Check table names and column names", "Verify query semantics"],
                    int((time.time() - start_time) * 1000)
                )

            # Format results with LLM summary
            tables_used = [t["table_name"] for t in selection["selected_tables"]]
            json_results, file_path, formatted_result = self.quality_inspector.format_response(
                sql, results, row_count, query, tables_used, database_id
            )

            # Build response
            execution_time = int((time.time() - start_time) * 1000)

            response = QueryResponse(
                status=QueryStatus.SUCCESS,
                query=query,
                tables_used=tables_used,
                tier=selection["tier"],
                row_count=row_count,
                result=json_results,
                csv_path=None,  # Deprecated - keeping for backward compatibility
                sql_generated=sql,
                joins=self._format_joins(selection.get("joins", [])),
                execution_time_ms=execution_time,
                confidence=selection.get("confidence", 0.5),
                formatted_result=formatted_result,
                file_path=file_path
            )

            logger.info(f"Query processed successfully in {execution_time}ms")
            return response

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
