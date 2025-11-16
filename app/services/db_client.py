"""
DuckDB client for SQL execution.
"""
import duckdb
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("db_client")


class DuckDBClient:
    """DuckDB client for SQL query execution."""

    def __init__(self):
        """Initialize DuckDB client."""
        self.connection = None
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to DuckDB database.

        Returns:
            True if connection successful
        """
        try:
            db_path = Path(settings.DUCKDB_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            self.connection = duckdb.connect(str(db_path))
            self.connected = True
            logger.info(f"Connected to DuckDB: {db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from DuckDB."""
        if self.connection:
            self.connection.close()
            self.connected = False
            logger.info("Disconnected from DuckDB")

    def execute_query(
        self,
        sql: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results as list of dicts.

        Args:
            sql: SQL query string
            limit: Optional row limit

        Returns:
            List of result rows as dictionaries
        """
        if not self.connected:
            raise ConnectionError("DuckDB not connected")

        try:
            # Add limit if specified
            if limit is not None and "LIMIT" not in sql.upper():
                sql = f"{sql.rstrip(';')} LIMIT {limit}"

            # Execute query
            result = self.connection.execute(sql).fetchall()

            # Get column names
            description = self.connection.description
            if description:
                columns = [desc[0] for desc in description]
            else:
                columns = []

            # Convert to list of dicts
            rows = []
            for row in result:
                row_dict = dict(zip(columns, row))
                rows.append(row_dict)

            logger.info(f"Query executed successfully: {len(rows)} rows returned")
            return rows

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def execute_count(self, sql: str) -> int:
        """
        Execute COUNT query to get total rows.

        Args:
            sql: SQL query string

        Returns:
            Total row count
        """
        if not self.connected:
            raise ConnectionError("DuckDB not connected")

        try:
            # Wrap query in COUNT(*)
            count_sql = f"SELECT COUNT(*) as total FROM ({sql.rstrip(';')}) as subquery"
            result = self.connection.execute(count_sql).fetchone()
            count = result[0] if result else 0

            logger.info(f"Count query executed: {count} total rows")
            return count

        except Exception as e:
            logger.error(f"Error executing count query: {e}")
            raise

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL by executing with LIMIT 0.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to execute with LIMIT 0
            test_sql = f"{sql.rstrip(';')} LIMIT 0"
            self.connection.execute(test_sql)
            return True, None

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"SQL validation failed: {error_msg}")
            return False, error_msg

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of column metadata
        """
        if not self.connected:
            raise ConnectionError("DuckDB not connected")

        try:
            # Get table schema using PRAGMA
            result = self.connection.execute(
                f"PRAGMA table_info('{table_name}')"
            ).fetchall()

            columns = []
            for row in result:
                column = {
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "primary_key": bool(row[5])
                }
                columns.append(column)

            logger.info(f"Schema fetched for table {table_name}: {len(columns)} columns")
            return columns

        except Exception as e:
            logger.error(f"Error fetching table schema: {e}")
            raise

    def get_sample_data(
        self,
        table_name: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get sample data from table.

        Args:
            table_name: Name of the table
            limit: Number of sample rows

        Returns:
            List of sample rows
        """
        sql = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(sql)


# Global singleton instance
_db_client = None


def get_db_client() -> DuckDBClient:
    """Get DuckDB client singleton."""
    global _db_client
    if _db_client is None:
        _db_client = DuckDBClient()
        _db_client.connect()
    return _db_client
