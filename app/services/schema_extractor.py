"""
Database schema extractor for reading table metadata from various databases.
"""
import pytds
from typing import List, Dict, Any, Optional, Tuple
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

            # Extract schema in batches and yield immediately (streaming, no memory accumulation)
            batch_size = 1000  # Process 1000 tables at a time
            total_tables = len(tables)
            total_extracted = 0

            for i in range(0, total_tables, batch_size):
                batch_end = min(i + batch_size, total_tables)
                batch_tables = tables[i:batch_end]
                batch_table_names = [t['TABLE_NAME'] for t in batch_tables]

                logger.info(f"Processing tables {i+1}-{batch_end} of {total_tables} ({int((batch_end/total_tables)*100)}%)")

                try:
                    # Use optimized batch schema extraction (4 queries instead of N*4)
                    batch_schemas = self._get_batch_table_schemas(cursor, batch_table_names)
                    total_extracted += len(batch_schemas)
                    logger.info(f"Batch complete: {len(batch_schemas)} tables extracted (total: {total_extracted}/{total_tables})")

                    # Yield batch immediately for processing (streaming)
                    yield batch_schemas

                except Exception as e:
                    logger.warning(f"Batch query failed, falling back to individual queries: {e}")
                    # Fallback to individual queries if batch fails
                    fallback_schemas = []
                    for table_row in batch_tables:
                        table_name = table_row['TABLE_NAME']
                        try:
                            table_schema = self._get_table_schema(cursor, table_name)
                            fallback_schemas.append(table_schema)
                        except Exception as e:
                            logger.warning(f"Failed to extract schema for table '{table_name}': {e}")
                            continue

                    total_extracted += len(fallback_schemas)
                    logger.info(f"Batch complete (fallback): {len(fallback_schemas)} tables extracted (total: {total_extracted}/{total_tables})")
                    yield fallback_schemas

            logger.info(f"Schema extraction complete: {total_extracted}/{total_tables} tables extracted successfully")

            cursor.close()
            self.connection.close()

        except Exception as e:
            logger.error(f"Error extracting MSSQL schemas: {e}")
            if self.connection:
                self.connection.close()
            raise

    def _get_batch_table_schemas(self, cursor, table_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed schemas for multiple tables in batch (optimized).

        Args:
            cursor: Database cursor
            table_names: List of table names to fetch

        Returns:
            List of table metadata dictionaries
        """
        if not table_names:
            return []

        # Create SQL IN clause with placeholders
        placeholders = ','.join(['%s'] * len(table_names))

        # 1. Get all columns for all tables in batch (1 query instead of N)
        columns_query = f"""
            SELECT
                TABLE_NAME,
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                CHARACTER_MAXIMUM_LENGTH,
                COLUMN_DEFAULT,
                ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME IN ({placeholders})
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        cursor.execute(columns_query, tuple(table_names))
        column_rows = cursor.fetchall()

        # Group columns by table
        columns_by_table = {}
        for row in column_rows:
            table_name = row[0]
            if table_name not in columns_by_table:
                columns_by_table[table_name] = []
            columns_by_table[table_name].append({
                'COLUMN_NAME': row[1],
                'DATA_TYPE': row[2],
                'IS_NULLABLE': row[3],
                'CHARACTER_MAXIMUM_LENGTH': row[4],
                'COLUMN_DEFAULT': row[5],
                'ORDINAL_POSITION': row[6]
            })

        # 2. Get all primary keys in batch (1 query instead of N)
        pk_query = f"""
            SELECT
                t.TABLE_NAME,
                COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            INNER JOIN INFORMATION_SCHEMA.TABLES t ON t.TABLE_NAME = kcu.TABLE_NAME
            WHERE OBJECTPROPERTY(OBJECT_ID(kcu.CONSTRAINT_SCHEMA + '.' + kcu.CONSTRAINT_NAME), 'IsPrimaryKey') = 1
            AND t.TABLE_NAME IN ({placeholders})
        """
        cursor.execute(pk_query, tuple(table_names))
        pk_rows = cursor.fetchall()

        # Group PKs by table
        pk_by_table = {}
        for row in pk_rows:
            table_name = row[0]
            if table_name not in pk_by_table:
                pk_by_table[table_name] = set()
            pk_by_table[table_name].add(row[1])

        # 3. Get all foreign keys in batch (1 query instead of N)
        fk_query = f"""
            SELECT
                OBJECT_NAME(fc.parent_object_id) AS TABLE_NAME,
                COL_NAME(fc.parent_object_id, fc.parent_column_id) AS COLUMN_NAME,
                OBJECT_NAME(fc.referenced_object_id) AS REFERENCED_TABLE,
                COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS REFERENCED_COLUMN
            FROM sys.foreign_key_columns AS fc
            INNER JOIN sys.foreign_keys AS f ON f.object_id = fc.constraint_object_id
            WHERE OBJECT_NAME(fc.parent_object_id) IN ({placeholders})
        """
        cursor.execute(fk_query, tuple(table_names))
        fk_rows = cursor.fetchall()

        # Group FKs by table
        fk_by_table = {}
        for row in fk_rows:
            table_name = row[0]
            column_name = row[1]
            if table_name not in fk_by_table:
                fk_by_table[table_name] = {}
            fk_by_table[table_name][column_name] = {
                'table': row[2],
                'column': row[3]
            }

        # 4. Get row counts for all tables in batch (1 query instead of N)
        count_query = f"""
            SELECT
                o.name AS TABLE_NAME,
                SUM(p.rows) AS row_count
            FROM sys.partitions p
            INNER JOIN sys.objects o ON p.object_id = o.object_id
            WHERE o.name IN ({placeholders})
            AND p.index_id IN (0, 1)
            GROUP BY o.name
        """
        cursor.execute(count_query, tuple(table_names))
        count_rows = cursor.fetchall()

        # Map row counts
        row_count_by_table = {}
        for row in count_rows:
            row_count_by_table[row[0]] = row[1] if row[1] else 0

        # 5. Get column descriptions for all tables in batch (1 query instead of N)
        desc_query = f"""
            SELECT
                t.name AS table_name,
                c.name AS column_name,
                CAST(ep.value AS NVARCHAR(MAX)) AS description
            FROM sys.tables t
            INNER JOIN sys.columns c ON c.object_id = t.object_id
            LEFT JOIN sys.extended_properties ep
                ON ep.major_id = t.object_id
                AND ep.minor_id = c.column_id
                AND ep.name = 'MS_Description'
            WHERE t.name IN ({placeholders})
            AND ep.value IS NOT NULL
            ORDER BY t.name, c.column_id
        """
        cursor.execute(desc_query, tuple(table_names))
        desc_rows = cursor.fetchall()

        # Map column descriptions
        desc_by_table = {}
        for row in desc_rows:
            table_name = row[0]
            column_name = row[1]
            description = row[2]
            if table_name not in desc_by_table:
                desc_by_table[table_name] = {}
            desc_by_table[table_name][column_name] = description

        # 6. Get table descriptions for all tables in batch (1 query instead of N)
        table_desc_query = f"""
            SELECT
                t.name AS table_name,
                CAST(ep.value AS NVARCHAR(MAX)) AS description
            FROM sys.tables t
            LEFT JOIN sys.extended_properties ep
                ON ep.major_id = t.object_id
                AND ep.minor_id = 0
                AND ep.name = 'MS_Description'
            WHERE t.name IN ({placeholders})
            AND ep.value IS NOT NULL
        """
        cursor.execute(table_desc_query, tuple(table_names))
        table_desc_rows = cursor.fetchall()

        # Map table descriptions
        table_desc_by_name = {}
        for row in table_desc_rows:
            table_desc_by_name[row[0]] = row[1]

        # Build table metadata for each table
        table_schemas = []
        for table_name in table_names:
            columns_data = columns_by_table.get(table_name, [])
            pk_columns = pk_by_table.get(table_name, set())
            fk_map = fk_by_table.get(table_name, {})
            row_count = row_count_by_table.get(table_name, 0)
            desc_map = desc_by_table.get(table_name, {})

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

                # Add column description if available
                if col['COLUMN_NAME'] in desc_map:
                    column_info['description'] = desc_map[col['COLUMN_NAME']]

                columns.append(column_info)

            # Build table metadata
            table_metadata = {
                'table_name': table_name,
                'name': table_name,
                'row_count': row_count,
                'columns': columns
            }

            # Add table description if available (user-provided from MSSQL extended properties)
            if table_name in table_desc_by_name:
                table_metadata['description'] = table_desc_by_name[table_name]

            table_schemas.append(table_metadata)

        logger.info(f"Batch extracted schemas for {len(table_schemas)} tables with 6 queries (optimized)")
        return table_schemas

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

        # Get table description (user-provided from MSSQL extended properties)
        table_desc_query = """
            SELECT CAST(ep.value AS NVARCHAR(MAX)) AS description
            FROM sys.tables t
            LEFT JOIN sys.extended_properties ep
                ON ep.major_id = t.object_id
                AND ep.minor_id = 0
                AND ep.name = 'MS_Description'
            WHERE t.name = %s
            AND ep.value IS NOT NULL
        """
        cursor.execute(table_desc_query, (table_name,))
        table_desc_result = cursor.fetchone()
        table_description = table_desc_result[0] if table_desc_result else None

        # Get column descriptions (user-provided from MSSQL extended properties)
        col_desc_query = """
            SELECT
                c.name AS column_name,
                CAST(ep.value AS NVARCHAR(MAX)) AS description
            FROM sys.tables t
            INNER JOIN sys.columns c ON c.object_id = t.object_id
            LEFT JOIN sys.extended_properties ep
                ON ep.major_id = t.object_id
                AND ep.minor_id = c.column_id
                AND ep.name = 'MS_Description'
            WHERE t.name = %s
            AND ep.value IS NOT NULL
        """
        cursor.execute(col_desc_query, (table_name,))
        col_desc_rows = cursor.fetchall()

        col_desc_map = {}
        for row in col_desc_rows:
            col_desc_map[row[0]] = row[1]

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

            # Add column description if available
            if col['COLUMN_NAME'] in col_desc_map:
                column_info['description'] = col_desc_map[col['COLUMN_NAME']]

            columns.append(column_info)

        # Build table metadata
        table_metadata = {
            'table_name': table_name,
            'name': table_name,  # Alias for compatibility
            'row_count': row_count,
            'columns': columns
        }

        # Add table description if available (user-provided from MSSQL extended properties)
        if table_description:
            table_metadata['description'] = table_description

        logger.debug(f"Extracted schema for table '{table_name}': {len(columns)} columns, ~{row_count} rows")

        return table_metadata

    def get_table_schema(self, table_name: str, database_name: str) -> Dict[str, Any]:
        """
        Get schema (full metadata) for a single table.

        Args:
            table_name: Name of the table
            database_name: Database name

        Returns:
            Table metadata dictionary with table_name, columns, row_count, etc.
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

            # Return just the columns list for backwards compatibility with query pipeline
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

    def write_table_descriptions_batch(
        self,
        database_name: str,
        table_descriptions: List[Dict[str, str]]
    ) -> Tuple[int, int, List[str]]:
        """
        Write table descriptions to MSSQL extended properties in batch.

        Args:
            database_name: Database name
            table_descriptions: List of dicts with 'table_name' and 'description' keys

        Returns:
            Tuple of (num_added, num_updated, errors)
        """
        if self.connection_type != 'mssql':
            raise NotImplementedError("Description writing only supported for MSSQL")

        try:
            # Decrypt password
            encrypted_password = self.config.get('password', '')
            password = decrypt_password(encrypted_password)

            # Connect to MSSQL
            logger.info(f"Connecting to MSSQL to write descriptions for {len(table_descriptions)} tables...")
            connection = pytds.connect(
                server=self.config.get('host'),
                port=self.config.get('port', 1433),
                user=self.config.get('username'),
                password=password,
                database=database_name,
                autocommit=True
            )

            cursor = connection.cursor()

            # Get table names that have descriptions to write
            tables_with_desc = [td for td in table_descriptions if td.get('description')]

            if not tables_with_desc:
                logger.info("No descriptions to write")
                cursor.close()
                connection.close()
                return 0, 0, []

            table_names = [td['table_name'] for td in tables_with_desc]

            # Check which tables already have extended properties
            placeholders = ','.join(['%s'] * len(table_names))
            check_query = f"""
                SELECT t.name AS table_name
                FROM sys.tables t
                INNER JOIN sys.extended_properties ep
                    ON ep.major_id = t.object_id
                    AND ep.minor_id = 0
                    AND ep.name = 'MS_Description'
                WHERE t.name IN ({placeholders})
            """

            cursor.execute(check_query, tuple(table_names))
            existing_tables = {row[0] for row in cursor.fetchall()}

            logger.info(f"Found {len(existing_tables)} tables with existing descriptions")

            num_added = 0
            num_updated = 0
            errors = []

            # Write each description
            for td in tables_with_desc:
                table_name = td['table_name']
                description = td['description']

                try:
                    # Escape single quotes for SQL
                    escaped_desc = description.replace("'", "''")

                    if table_name in existing_tables:
                        # Update existing
                        logger.info(f"[MSSQL WRITE] Updating MS_Description for table: {table_name}")
                        logger.info(f"[MSSQL WRITE] Description: {description[:100]}{'...' if len(description) > 100 else ''}")
                        query = f"""
                            EXEC sp_updateextendedproperty
                                @name = N'MS_Description',
                                @value = N'{escaped_desc}',
                                @level0type = N'SCHEMA', @level0name = 'dbo',
                                @level1type = N'TABLE', @level1name = N'{table_name}'
                        """
                        cursor.execute(query)
                        num_updated += 1
                        logger.info(f"[MSSQL WRITE] ✓ Successfully updated MS_Description for {table_name}")
                    else:
                        # Add new
                        logger.info(f"[MSSQL WRITE] Adding MS_Description for table: {table_name}")
                        logger.info(f"[MSSQL WRITE] Description: {description[:100]}{'...' if len(description) > 100 else ''}")
                        query = f"""
                            EXEC sp_addextendedproperty
                                @name = N'MS_Description',
                                @value = N'{escaped_desc}',
                                @level0type = N'SCHEMA', @level0name = 'dbo',
                                @level1type = N'TABLE', @level1name = N'{table_name}'
                        """
                        cursor.execute(query)
                        num_added += 1
                        logger.info(f"[MSSQL WRITE] ✓ Successfully added MS_Description for {table_name}")

                except Exception as e:
                    error_msg = f"Failed to write description for {table_name}: {str(e)}"
                    logger.error(f"[MSSQL WRITE] ✗ {error_msg}")
                    errors.append(error_msg)

            cursor.close()
            connection.close()

            logger.info(f"Batch write complete: {num_added} added, {num_updated} updated, {len(errors)} errors")
            return num_added, num_updated, errors

        except Exception as e:
            logger.error(f"Error writing descriptions to MSSQL: {e}")
            raise

    def write_field_descriptions_batch(
        self,
        database_name: str,
        table_field_descriptions: List[Dict[str, Any]]
    ) -> Tuple[int, int, List[str]]:
        """
        Write field/column descriptions to MSSQL extended properties in batch.

        Args:
            database_name: Database name
            table_field_descriptions: List of dicts with 'table_name' and 'field_descriptions' keys
                field_descriptions is a list of dicts with 'field_name' and 'description'

        Returns:
            Tuple of (num_added, num_updated, errors)
        """
        if self.connection_type != 'mssql':
            raise NotImplementedError("Description writing only supported for MSSQL")

        try:
            # Decrypt password
            encrypted_password = self.config.get('password', '')
            password = decrypt_password(encrypted_password)

            # Connect to MSSQL
            logger.info(f"Connecting to MSSQL to write field descriptions...")
            connection = pytds.connect(
                server=self.config.get('host'),
                port=self.config.get('port', 1433),
                user=self.config.get('username'),
                password=password,
                database=database_name,
                autocommit=True
            )

            cursor = connection.cursor()

            # Collect all table-column pairs that have descriptions
            field_writes = []
            for td in table_field_descriptions:
                table_name = td['table_name']
                field_descs = td.get('field_descriptions', [])
                if field_descs:
                    for fd in field_descs:
                        field_name = fd.get('field_name')
                        description = fd.get('description')
                        if field_name and description:
                            field_writes.append({
                                'table_name': table_name,
                                'field_name': field_name,
                                'description': description
                            })

            if not field_writes:
                logger.info("No field descriptions to write")
                cursor.close()
                connection.close()
                return 0, 0, []

            logger.info(f"Writing {len(field_writes)} field descriptions...")

            # Get existing column descriptions
            table_names = list(set(fw['table_name'] for fw in field_writes))
            placeholders = ','.join(['%s'] * len(table_names))
            check_query = f"""
                SELECT
                    t.name AS table_name,
                    c.name AS column_name
                FROM sys.tables t
                INNER JOIN sys.columns c ON c.object_id = t.object_id
                INNER JOIN sys.extended_properties ep
                    ON ep.major_id = t.object_id
                    AND ep.minor_id = c.column_id
                    AND ep.name = 'MS_Description'
                WHERE t.name IN ({placeholders})
            """

            cursor.execute(check_query, tuple(table_names))
            existing_columns = {(row[0], row[1]) for row in cursor.fetchall()}

            logger.info(f"Found {len(existing_columns)} columns with existing descriptions")

            num_added = 0
            num_updated = 0
            errors = []

            # Write each field description
            for fw in field_writes:
                table_name = fw['table_name']
                field_name = fw['field_name']
                description = fw['description']

                try:
                    # Escape single quotes for SQL
                    escaped_desc = description.replace("'", "''")

                    if (table_name, field_name) in existing_columns:
                        # Update existing
                        logger.debug(f"[MSSQL WRITE] Updating MS_Description for {table_name}.{field_name}")
                        query = f"""
                            EXEC sp_updateextendedproperty
                                @name = N'MS_Description',
                                @value = N'{escaped_desc}',
                                @level0type = N'SCHEMA', @level0name = 'dbo',
                                @level1type = N'TABLE', @level1name = N'{table_name}',
                                @level2type = N'COLUMN', @level2name = N'{field_name}'
                        """
                        cursor.execute(query)
                        num_updated += 1
                    else:
                        # Add new
                        logger.debug(f"[MSSQL WRITE] Adding MS_Description for {table_name}.{field_name}")
                        query = f"""
                            EXEC sp_addextendedproperty
                                @name = N'MS_Description',
                                @value = N'{escaped_desc}',
                                @level0type = N'SCHEMA', @level0name = 'dbo',
                                @level1type = N'TABLE', @level1name = N'{table_name}',
                                @level2type = N'COLUMN', @level2name = N'{field_name}'
                        """
                        cursor.execute(query)
                        num_added += 1

                except Exception as e:
                    error_msg = f"Failed to write description for {table_name}.{field_name}: {str(e)}"
                    logger.error(f"[MSSQL WRITE] ✗ {error_msg}")
                    errors.append(error_msg)

            cursor.close()
            connection.close()

            logger.info(f"Field descriptions write complete: {num_added} added, {num_updated} updated, {len(errors)} errors")
            return num_added, num_updated, errors

        except Exception as e:
            logger.error(f"Error writing field descriptions to MSSQL: {e}")
            raise

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
