"""
Quality Inspector: Validate and execute SQL queries.
Stages 6 & 7 of the pipeline.
"""
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import io
from openai import OpenAI

from app.services import get_mongo_client, get_minio_client
from app.services.schema_extractor import SchemaExtractor
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("quality_inspector")


class QualityInspector:
    """Agent for SQL validation, execution, and result formatting."""

    def __init__(self):
        """Initialize Quality Inspector."""
        pass

    def validate_and_execute(
        self,
        sql: str,
        database_id: str
    ) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Any]]], Optional[int]]:
        """
        Validate SQL and execute if valid against the actual MSSQL database.

        Args:
            sql: SQL query string
            database_id: MongoDB database ID

        Returns:
            Tuple of (is_valid, error_message, results, row_count)
        """
        logger.info("Validating SQL...")

        # Get connection config for this database
        mongo_client = get_mongo_client()
        config_data = mongo_client.get_database_connection_config(database_id)
        database_name = config_data["database_name"]
        connection_config = config_data["connection_config"]

        # Create schema extractor with connection config
        extractor = SchemaExtractor(connection_config)

        try:
            # Validate with dry run
            is_valid, error_msg = extractor.validate_sql(sql, database_name)

            if not is_valid:
                logger.warning(f"SQL validation failed: {error_msg}")

                # Attempt auto-repair
                repaired_sql = self._auto_repair(sql, error_msg)
                if repaired_sql:
                    logger.info("Auto-repair attempted, re-validating...")
                    is_valid, error_msg = extractor.validate_sql(repaired_sql, database_name)
                    if is_valid:
                        sql = repaired_sql
                        logger.info("Auto-repair successful!")

            if not is_valid:
                extractor.close()
                return False, error_msg, None, None

            # Execute query
            # Get count first
            row_count = extractor.execute_count(sql, database_name)
            logger.info(f"Query will return {row_count} rows")

            # Execute with appropriate limit
            if row_count <= settings.MAX_RESULT_ROWS_IN_RESPONSE:
                # Small result set - return as JSON
                results = extractor.execute_query(sql, database_name)
                extractor.close()
                return True, None, results, row_count
            else:
                # Large result set - will export to CSV
                results = extractor.execute_query(sql, database_name)  # Get all rows
                extractor.close()
                return True, None, results, row_count

        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            extractor.close()
            return False, str(e), None, None

    def format_response(
        self,
        sql: str,
        results: List[Dict[str, Any]],
        row_count: int,
        query: str,
        tables_used: List[str],
        database_id: str
    ) -> Tuple[Optional[List[Dict]], Optional[str], Optional[str]]:
        """
        Format results with LLM summary and optional Parquet export to MinIO.

        Args:
            sql: Executed SQL query
            results: Query results
            row_count: Total row count
            query: Original natural language query
            tables_used: List of tables used in query
            database_id: MongoDB database ID for file naming

        Returns:
            Tuple of (json_results, file_path, formatted_result)
        """
        file_path = None
        json_results = None
        formatted_result = None

        if row_count <= settings.MAX_RESULT_ROWS_IN_RESPONSE:
            # Small result set - return as JSON and generate summary
            logger.info(f"Returning {row_count} rows as JSON")
            json_results = results

            # Generate natural language summary with top 5 rows
            formatted_result = self._generate_natural_language_summary(
                query, tables_used, row_count, results[:5], None
            )
        else:
            # Large result set - export to Parquet in MinIO
            file_path = self._export_to_parquet_minio(results, database_id)
            logger.info(f"Exported {row_count} rows to MinIO: {file_path}")

            # Generate natural language summary with top 5 rows and file info
            formatted_result = self._generate_natural_language_summary(
                query, tables_used, row_count, results[:5], file_path
            )

        return json_results, file_path, formatted_result

    def _auto_repair(self, sql: str, error: str) -> Optional[str]:
        """
        Attempt to auto-repair SQL based on error.

        Args:
            sql: Original SQL
            error: Error message

        Returns:
            Repaired SQL or None
        """
        repaired = sql

        # Fix: Convert 'active'/'inactive' to status codes
        if "could not convert" in error.lower() and "active" in sql.lower():
            logger.info("Auto-repair: Converting 'active'/'inactive' to status codes")
            repaired = repaired.replace("= 'active'", "= 1")
            repaired = repaired.replace("= 'inactive'", "= 0")
            return repaired

        # Fix: Remove trailing semicolons before LIMIT
        if "syntax error near limit" in error.lower():
            logger.info("Auto-repair: Removing semicolon before LIMIT")
            repaired = repaired.replace("; LIMIT", " LIMIT")
            return repaired

        # Add more auto-repair rules here

        return None

    def _export_to_parquet_minio(self, results: List[Dict[str, Any]], database_id: str) -> str:
        """
        Export results to Parquet format and upload to MinIO.

        Args:
            results: Query results
            database_id: Database ID for file naming

        Returns:
            MinIO file path (s3:// format)
        """
        try:
            # Upload to MinIO using MinIOClient
            minio_client = get_minio_client()
            object_path = minio_client.upload_parquet(
                data=results,
                query="query_results",  # Generic query name
                database_id=database_id
            )

            # Return S3-style path
            minio_path = f"s3://{settings.MINIO_BUCKET}/{object_path}"
            logger.info(f"Exported to MinIO: {minio_path}")
            return minio_path

        except Exception as e:
            logger.error(f"Error exporting to Parquet/MinIO: {e}")
            raise

    def _generate_natural_language_summary(
        self,
        query: str,
        tables_used: List[str],
        row_count: int,
        top_rows: List[Dict[str, Any]],
        file_path: Optional[str]
    ) -> str:
        """
        Generate natural language summary of query results using LLM.

        Args:
            query: Original natural language query
            tables_used: List of tables used
            row_count: Total number of rows
            top_rows: First 5 rows of results
            file_path: MinIO file path if large result set

        Returns:
            Natural language summary
        """
        try:
            # Prepare context for LLM
            context = f"""
Query: "{query}"

Tables involved: {', '.join(tables_used)}
Total rows returned: {row_count:,}
"""

            if file_path:
                context += f"\nFull results stored at: {file_path}"

            if top_rows:
                context += f"\n\nFirst {len(top_rows)} rows (sample):\n"
                for i, row in enumerate(top_rows, 1):
                    context += f"{i}. {row}\n"

            # Call LLM to generate summary
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Summarize query results in natural language, highlighting key insights and metadata."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize these query results:\n\n{context}"
                    }
                ],
                temperature=0.3,
                max_tokens=300
            )

            summary = response.choices[0].message.content.strip()
            logger.info("Generated natural language summary")
            return summary

        except Exception as e:
            logger.error(f"Error generating LLM summary: {e}")
            # Fallback to basic summary
            summary = f"Query returned {row_count:,} rows from {len(tables_used)} table(s): {', '.join(tables_used)}."
            if file_path:
                summary += f" Full results available at: {file_path}"
            return summary


def create_quality_inspector() -> QualityInspector:
    """Factory function to create Quality Inspector instance."""
    return QualityInspector()
