"""
Database schema extractor for reading table metadata from various databases.
"""
import pytds
from typing import List, Dict, Any, Optional
from app.utils.logger import setup_logger
from app.utils.encryption import decrypt_password

logger = setup_logger("schema_extractor")


class SchemaExtractor:
    """Extract table schemas from databases."""

    def __init__(self, connection_config: Dict[str, Any]):
        """
        Initialize schema extractor.

        Args:
            connection_config: Connection configuration with keys:
                - connectionType: 'mssql', 'mysql', 'postgresql', etc.
                - host, port, username, password (encrypted)
                - database (optional)
        """
        self.config = connection_config
        self.connection_type = connection_config.get('connectionType', '').lower()
        self.connection = None

    def extract_schemas(self, database_name: str) -> List[Dict[str, Any]]:
        """
        Extract all table schemas from database.

        Args:
            database_name: Name of the database to extract from

        Returns:
            List of table metadata dictionaries
        """
        if self.connection_type == 'mssql':
            return self._extract_mssql_schemas(database_name)
        else:
            raise NotImplementedError(f"Database type '{self.connection_type}' not yet supported")

    def _extract_mssql_schemas(self, database_name: str) -> List[Dict[str, Any]]:
        """
        Extract schemas from MSSQL database.

        Args:
            database_name: Database name

        Returns:
            List of table metadata
        """
        try:
            # Decrypt password
            encrypted_password = self.config.get('password', '')
            password = decrypt_password(encrypted_password)

            # Connect to MSSQL
            logger.info(f"Connecting to MSSQL: {self.config.get('host')}:{self.config.get('port')}")
            self.connection = pytds.connect(
                server=self.config.get('host'),
                port=self.config.get('port', 1433),
                user=self.config.get('username'),
                password=password,
                database=database_name,
                autocommit=True
            )

            cursor = self.connection.cursor()

            # Get all tables
            tables_query = """
                SELECT TABLE_NAME, TABLE_TYPE
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                AND TABLE_SCHEMA = 'dbo'
                ORDER BY TABLE_NAME
            """
            cursor.execute(tables_query)
            table_rows = cursor.fetchall()

            # Convert rows to dicts (pytds returns tuples)
            tables = []
            for row in table_rows:
                tables.append({
                    'TABLE_NAME': row[0],
                    'TABLE_TYPE': row[1]
                })

            logger.info(f"Found {len(tables)} tables in database '{database_name}'")

            # Extract schema for each table
            table_schemas = []
            for table_row in tables:
                table_name = table_row['TABLE_NAME']
                table_schema = self._get_table_schema(cursor, table_name)
                table_schemas.append(table_schema)

            cursor.close()
            self.connection.close()

            return table_schemas

        except Exception as e:
            logger.error(f"Error extracting MSSQL schemas: {e}")
            if self.connection:
                self.connection.close()
            raise

    def _get_table_schema(self, cursor, table_name: str) -> Dict[str, Any]:
        """
        Get detailed schema for a single table.

        Args:
            cursor: Database cursor
            table_name: Name of the table

        Returns:
            Table metadata dictionary
        """
        # Get columns
        columns_query = """
            SELECT
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                CHARACTER_MAXIMUM_LENGTH,
                COLUMN_DEFAULT,
                ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """
        cursor.execute(columns_query, (table_name,))
        column_rows = cursor.fetchall()

        # Convert to dicts
        columns_data = []
        for row in column_rows:
            columns_data.append({
                'COLUMN_NAME': row[0],
                'DATA_TYPE': row[1],
                'IS_NULLABLE': row[2],
                'CHARACTER_MAXIMUM_LENGTH': row[3],
                'COLUMN_DEFAULT': row[4],
                'ORDINAL_POSITION': row[5]
            })

        # Get primary keys
        pk_query = """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
            AND TABLE_NAME = %s
        """
        cursor.execute(pk_query, (table_name,))
        pk_rows = cursor.fetchall()
        pk_columns = {row[0] for row in pk_rows}

        # Get foreign keys
        fk_query = """
            SELECT
                COL_NAME(fc.parent_object_id, fc.parent_column_id) AS COLUMN_NAME,
                OBJECT_NAME(fc.referenced_object_id) AS REFERENCED_TABLE,
                COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS REFERENCED_COLUMN
            FROM sys.foreign_key_columns AS fc
            INNER JOIN sys.foreign_keys AS f ON f.object_id = fc.constraint_object_id
            WHERE OBJECT_NAME(fc.parent_object_id) = %s
        """
        cursor.execute(fk_query, (table_name,))
        fk_rows = cursor.fetchall()

        fk_map = {}
        for row in fk_rows:
            fk_map[row[0]] = {
                'table': row[1],
                'column': row[2]
            }

        # Get row count estimate
        count_query = f"""
            SELECT SUM(p.rows) AS row_count
            FROM sys.partitions p
            INNER JOIN sys.objects o ON p.object_id = o.object_id
            WHERE o.name = %s AND p.index_id IN (0, 1)
        """
        cursor.execute(count_query, (table_name,))
        count_result = cursor.fetchone()
        row_count = count_result[0] if count_result and count_result[0] else 0

        # Format columns
        columns = []
        for col in columns_data:
            column_info = {
                'name': col['COLUMN_NAME'],
                'type': col['DATA_TYPE'].upper(),
                'nullable': col['IS_NULLABLE'] == 'YES',
                'is_primary_key': col['COLUMN_NAME'] in pk_columns
            }

            if col['COLUMN_NAME'] in fk_map:
                column_info['is_foreign_key'] = True
                fk_ref = fk_map[col['COLUMN_NAME']]
                column_info['references'] = f"{fk_ref['table']}.{fk_ref['column']}"

            if col['CHARACTER_MAXIMUM_LENGTH']:
                column_info['max_length'] = col['CHARACTER_MAXIMUM_LENGTH']

            if col['COLUMN_DEFAULT']:
                column_info['default'] = col['COLUMN_DEFAULT']

            columns.append(column_info)

        # Build table metadata
        table_metadata = {
            'table_name': table_name,
            'name': table_name,  # Alias for compatibility
            'row_count': row_count,
            'columns': columns
        }

        logger.debug(f"Extracted schema for table '{table_name}': {len(columns)} columns, ~{row_count} rows")

        return table_metadata

    def get_table_schema(self, table_name: str, database_name: str) -> List[Dict[str, Any]]:
        """
        Get schema (columns) for a single table.

        Args:
            table_name: Name of the table
            database_name: Database name

        Returns:
            List of column dictionaries
        """
        try:
            # Create temporary connection for schema fetch
            if not self.connection:
                encrypted_password = self.config.get('password', '')
                password = decrypt_password(encrypted_password)

                self.connection = pytds.connect(
                    server=self.config.get('host'),
                    port=self.config.get('port', 1433),
                    user=self.config.get('username'),
                    password=password,
                    database=database_name,
                    autocommit=True
                )

            cursor = self.connection.cursor()
            table_metadata = self._get_table_schema(cursor, table_name)
            cursor.close()

            return table_metadata.get('columns', [])

        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return []

    def get_row_count(self, table_name: str, database_name: str) -> int:
        """
        Get approximate row count for a table.

        Args:
            table_name: Name of the table
            database_name: Database name

        Returns:
            Approximate row count
        """
        try:
            # Create temporary connection for row count
            if not self.connection:
                encrypted_password = self.config.get('password', '')
                password = decrypt_password(encrypted_password)

                self.connection = pytds.connect(
                    server=self.config.get('host'),
                    port=self.config.get('port', 1433),
                    user=self.config.get('username'),
                    password=password,
                    database=database_name,
                    autocommit=True
                )

            cursor = self.connection.cursor()
            table_metadata = self._get_table_schema(cursor, table_name)
            cursor.close()

            return table_metadata.get('row_count', 0)

        except Exception as e:
            logger.error(f"Error getting row count for table {table_name}: {e}")
            return 0

    def execute_query(self, sql: str, database_name: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query string
            database_name: Database name

        Returns:
            List of result dictionaries
        """
        try:
            # Connect if not already connected
            if not self.connection:
                encrypted_password = self.config.get('password', '')
                password = decrypt_password(encrypted_password)

                self.connection = pytds.connect(
                    server=self.config.get('host'),
                    port=self.config.get('port', 1433),
                    user=self.config.get('username'),
                    password=password,
                    database=database_name,
                    autocommit=True
                )

            cursor = self.connection.cursor()
            cursor.execute(sql)

            # Fetch all rows
            rows = cursor.fetchall()

            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Convert to list of dicts
            results = []
            for row in rows:
                result_dict = {}
                for i, col_name in enumerate(columns):
                    result_dict[col_name] = row[i]
                results.append(result_dict)

            cursor.close()
            return results

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def execute_count(self, sql: str, database_name: str) -> int:
        """
        Execute SQL and return row count.

        Args:
            sql: SQL query string
            database_name: Database name

        Returns:
            Number of rows
        """
        try:
            # Wrap SQL in COUNT query
            count_sql = f"SELECT COUNT(*) as count FROM ({sql}) as subquery"
            results = self.execute_query(count_sql, database_name)
            return results[0]['count'] if results else 0
        except Exception as e:
            logger.error(f"Error executing count: {e}")
            # Fallback: execute query and count results
            results = self.execute_query(sql, database_name)
            return len(results)

    def validate_sql(self, sql: str, database_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL by attempting to execute it with LIMIT 0.

        Args:
            sql: SQL query string
            database_name: Database name

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to execute with TOP 0 (MSSQL syntax)
            validation_sql = f"SELECT TOP 0 * FROM ({sql}) as validation_query"
            self.execute_query(validation_sql, database_name)
            return True, None
        except Exception as e:
            return False, str(e)

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def _infer_category(self, table_name: str) -> str:
        """
        Infer table category from name.

        Args:
            table_name: Name of the table

        Returns:
            Category string
        """
        table_lower = table_name.lower()

        categories = {
            'employee': ['employee', 'emp', 'staff', 'worker'],
            'customer': ['customer', 'client', 'cust'],
            'sales': ['sales', 'order', 'invoice', 'transaction'],
            'product': ['product', 'item', 'inventory'],
            'department': ['department', 'dept', 'division'],
            'finance': ['payment', 'billing', 'account'],
            'log': ['log', 'audit', 'history']
        }

        for category, keywords in categories.items():
            if any(keyword in table_lower for keyword in keywords):
                return category

        return 'general'
