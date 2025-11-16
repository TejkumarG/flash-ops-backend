# Adding Database Support

## Overview

Flash-Ops is designed to be database-agnostic. Currently supports **MSSQL**, with MySQL and PostgreSQL support requiring minimal implementation.

## Architecture for Database Support

### Database-Agnostic Components (No Changes Needed)

These components work with ANY database type:

1. **Vector Store** (`app/services/vector_store.py`)
   - Semantic search on table names/schemas
   - Database-independent

2. **Table Clustering** (`app/agents/table_clustering.py`)
   - Groups similar tables
   - Works on embeddings, not SQL

3. **Table Selection** (`app/agents/table_selector.py`)
   - Selects optimal tables
   - Database-independent logic

4. **Schema Packager** (`app/agents/schema_packager.py`)
   - Formats schema for LLM
   - Generic markdown format

5. **SQL Generator** (`app/agents/sql_generator.py`)
   - LLM adapts to any SQL dialect
   - Prompts are database-agnostic

6. **MinIO Client** (`app/services/minio_client.py`)
   - Stores results in Parquet
   - Database-independent

7. **Quality Inspector** (partial)
   - Result formatting and LLM summary are database-agnostic
   - Only validation/execution are database-specific

### Database-Specific Component (Requires Implementation)

**SchemaExtractor** (`app/services/schema_extractor.py`)
- The ONLY component that needs database-specific code
- Handles: connection, schema extraction, SQL validation, query execution

## Step-by-Step: Adding MySQL Support

### 1. Install MySQL Connector

```bash
pip install mysql-connector-python
```

Update `requirements.txt`:
```
mysql-connector-python==8.2.0
```

### 2. Implement MySQL Methods in SchemaExtractor

**File**: `app/services/schema_extractor.py`

```python
def extract_schemas(self, database_name: str) -> List[Dict[str, Any]]:
    """Extract all table schemas from database."""
    if self.connection_type == 'mssql':
        return self._extract_mssql_schemas(database_name)
    elif self.connection_type == 'mysql':
        return self._extract_mysql_schemas(database_name)  # NEW
    else:
        raise NotImplementedError(f"Database type '{self.connection_type}' not yet supported")

def _extract_mysql_schemas(self, database_name: str) -> List[Dict[str, Any]]:
    """Extract schemas from MySQL database."""
    try:
        import mysql.connector

        # Decrypt password
        encrypted_password = self.config.get('password', '')
        password = decrypt_password(encrypted_password)

        # Connect to MySQL
        logger.info(f"Connecting to MySQL: {self.config.get('host')}:{self.config.get('port')}")
        self.connection = mysql.connector.connect(
            host=self.config.get('host'),
            port=self.config.get('port', 3306),
            user=self.config.get('username'),
            password=password,
            database=database_name
        )

        cursor = self.connection.cursor(dictionary=True)

        # Get all tables
        tables_query = """
            SELECT TABLE_NAME, TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = %s
            AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """
        cursor.execute(tables_query, (database_name,))
        tables = cursor.fetchall()

        logger.info(f"Found {len(tables)} tables in database '{database_name}'")

        # Extract schema for each table
        table_schemas = []
        for table_row in tables:
            table_name = table_row['TABLE_NAME']
            table_schema = self._get_mysql_table_schema(cursor, table_name, database_name)
            table_schemas.append(table_schema)

        cursor.close()
        return table_schemas

    except Exception as e:
        logger.error(f"Error extracting MySQL schemas: {e}")
        raise

def _get_mysql_table_schema(self, cursor, table_name: str, database_name: str) -> Dict[str, Any]:
    """Get detailed schema for a MySQL table."""
    # Get columns
    columns_query = """
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_KEY,
            EXTRA
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
    """
    cursor.execute(columns_query, (database_name, table_name))
    columns = cursor.fetchall()

    # Get foreign keys
    fk_query = """
        SELECT
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = %s
        AND REFERENCED_TABLE_NAME IS NOT NULL
    """
    cursor.execute(fk_query, (database_name, table_name))
    foreign_keys = cursor.fetchall()

    return {
        'table_name': table_name,
        'columns': [
            {
                'name': col['COLUMN_NAME'],
                'type': col['DATA_TYPE'],
                'nullable': col['IS_NULLABLE'] == 'YES',
                'primary_key': col['COLUMN_KEY'] == 'PRI',
                'auto_increment': 'auto_increment' in col['EXTRA'].lower()
            }
            for col in columns
        ],
        'foreign_keys': [
            {
                'column': fk['COLUMN_NAME'],
                'referenced_table': fk['REFERENCED_TABLE_NAME'],
                'referenced_column': fk['REFERENCED_COLUMN_NAME']
            }
            for fk in foreign_keys
        ]
    }

def validate_sql(self, sql: str, database_name: str) -> Tuple[bool, Optional[str]]:
    """Validate SQL syntax with dry run."""
    try:
        if self.connection_type == 'mssql':
            return self._validate_mssql_sql(sql, database_name)
        elif self.connection_type == 'mysql':
            return self._validate_mysql_sql(sql, database_name)  # NEW
        else:
            raise NotImplementedError(f"Validation not implemented for {self.connection_type}")

    except Exception as e:
        return False, str(e)

def _validate_mysql_sql(self, sql: str, database_name: str) -> Tuple[bool, Optional[str]]:
    """Validate MySQL SQL using EXPLAIN."""
    try:
        if not self.connection:
            raise ConnectionError("Not connected to MySQL")

        cursor = self.connection.cursor()

        # Use EXPLAIN to validate without executing
        explain_sql = f"EXPLAIN {sql}"
        cursor.execute(explain_sql)
        cursor.fetchall()  # Consume results
        cursor.close()

        return True, None

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"MySQL SQL validation failed: {error_msg}")
        return False, error_msg

def execute_query(self, sql: str, database_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Execute query and return results."""
    if self.connection_type == 'mssql':
        return self._execute_mssql_query(sql, database_name, limit)
    elif self.connection_type == 'mysql':
        return self._execute_mysql_query(sql, database_name, limit)  # NEW
    else:
        raise NotImplementedError(f"Execution not implemented for {self.connection_type}")

def _execute_mysql_query(self, sql: str, database_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Execute MySQL query and return results."""
    try:
        if not self.connection:
            raise ConnectionError("Not connected to MySQL")

        cursor = self.connection.cursor(dictionary=True)

        # Add LIMIT if specified
        if limit:
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()

        logger.info(f"MySQL query executed successfully, returned {len(results)} rows")
        return results

    except Exception as e:
        logger.error(f"Error executing MySQL query: {e}")
        raise

def execute_count(self, sql: str, database_name: str) -> int:
    """Get count of results without fetching all rows."""
    if self.connection_type == 'mssql':
        return self._execute_mssql_count(sql, database_name)
    elif self.connection_type == 'mysql':
        return self._execute_mysql_count(sql, database_name)  # NEW
    else:
        raise NotImplementedError(f"Count not implemented for {self.connection_type}")

def _execute_mysql_count(self, sql: str, database_name: str) -> int:
    """Get count from MySQL query."""
    try:
        if not self.connection:
            raise ConnectionError("Not connected to MySQL")

        cursor = self.connection.cursor()

        # Wrap query in COUNT(*)
        count_sql = f"SELECT COUNT(*) as count FROM ({sql.rstrip(';')}) AS subquery"
        cursor.execute(count_sql)
        result = cursor.fetchone()
        cursor.close()

        return result[0] if result else 0

    except Exception as e:
        logger.error(f"Error getting MySQL count: {e}")
        raise
```

### 3. Register MySQL Database in MongoDB

```python
# Example registration
import requests

response = requests.post('http://localhost:8000/api/v1/databases/register', json={
    "database_name": "my_mysql_db",
    "connection_type": "mysql",  # KEY: Specify MySQL
    "host": "mysql.example.com",
    "port": 3306,
    "username": "db_user",
    "password": "db_password",
    "description": "Production MySQL database"
})
```

MongoDB document structure:
```json
{
  "_id": "...",
  "database_name": "my_mysql_db",
  "connection_config": {
    "connectionType": "mysql",
    "host": "mysql.example.com",
    "port": 3306,
    "username": "db_user",
    "password": "<encrypted>"
  },
  "created_at": "2025-01-17T...",
  "indexed": false
}
```

### 4. Generate Embeddings

```bash
curl -X POST 'http://localhost:8000/api/v1/embeddings/generate/<database_id>'
```

This will:
1. Connect to MySQL using `_extract_mysql_schemas()`
2. Extract table schemas from `INFORMATION_SCHEMA`
3. Generate embeddings
4. Store in Milvus

### 5. Query the Database

```bash
curl -X POST 'http://localhost:8000/api/v1/query/team' \
  -H 'Content-Type: application/json' \
  -d '{
    "api_key": "your_api_key",
    "query": "Show me all users created in the last 7 days"
  }'
```

**Everything works automatically!**
- Vector search finds relevant MySQL tables
- LLM generates MySQL-compatible SQL
- `_validate_mysql_sql()` validates syntax
- `_execute_mysql_query()` runs the query
- Results formatted and returned

## Adding PostgreSQL Support

Follow the same pattern:

```python
# 1. Install connector
pip install psycopg2-binary

# 2. Implement methods
def _extract_postgresql_schemas(self, database_name: str):
    import psycopg2
    # Similar to MySQL, use INFORMATION_SCHEMA

def _validate_postgresql_sql(self, sql: str, database_name: str):
    # Use EXPLAIN (same as MySQL)

def _execute_postgresql_query(self, sql: str, database_name: str, limit: Optional[int] = None):
    # Execute with psycopg2 cursor

# 3. Update factory methods in SchemaExtractor
elif self.connection_type == 'postgresql':
    return self._extract_postgresql_schemas(database_name)
```

## SQL Dialect Differences

The LLM (GPT-4) handles SQL dialect differences automatically:

| Feature | MSSQL | MySQL | PostgreSQL |
|---------|-------|-------|-----------|
| Limit | `TOP N` | `LIMIT N` | `LIMIT N` |
| String concat | `+` | `CONCAT()` | `\|\|` |
| Auto-increment | `IDENTITY` | `AUTO_INCREMENT` | `SERIAL` |
| Date format | `GETDATE()` | `NOW()` | `NOW()` |

**The LLM adapts based on schema context!** No manual SQL translation needed.

## Testing New Database Support

### 1. Unit Tests

```python
# tests/test_schema_extractor.py
def test_mysql_connection():
    config = {
        'connectionType': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'username': 'test',
        'password': encrypt_password('test')
    }
    extractor = SchemaExtractor(config)
    schemas = extractor.extract_schemas('test_db')
    assert len(schemas) > 0

def test_mysql_query_execution():
    # Test SQL execution
    results = extractor.execute_query("SELECT * FROM users LIMIT 5", 'test_db')
    assert isinstance(results, list)
```

### 2. Integration Tests

```python
# Test full pipeline with MySQL
response = client.post('/api/v1/query/team', json={
    'api_key': test_api_key,
    'query': 'Show me all active users'
})
assert response.status_code == 200
assert response.json()['status'] == 'success'
```

## Database Support Checklist

When adding a new database:

- [ ] Install database connector library
- [ ] Implement `_extract_<db>_schemas()` method
- [ ] Implement `_get_<db>_table_schema()` helper
- [ ] Implement `_validate_<db>_sql()` method
- [ ] Implement `_execute_<db>_query()` method
- [ ] Implement `_execute_<db>_count()` method
- [ ] Update factory methods in `extract_schemas()`, `validate_sql()`, `execute_query()`, `execute_count()`
- [ ] Add unit tests for connection and extraction
- [ ] Add integration tests for query execution
- [ ] Update documentation
- [ ] Test with real queries

## Connection Pooling (Future Enhancement)

For production, implement connection pooling:

```python
# app/services/connection_pool.py
class DatabaseConnectionPool:
    def __init__(self, config: Dict, pool_size: int = 5):
        self.config = config
        self.pool = self._create_pool(pool_size)

    def _create_pool(self, size: int):
        if self.config['connectionType'] == 'mysql':
            import mysql.connector.pooling
            return mysql.connector.pooling.MySQLConnectionPool(
                pool_name="flash_ops_pool",
                pool_size=size,
                **self.config
            )
```

## Summary

**Key Points:**
1. Only `SchemaExtractor` needs database-specific code
2. All other components are database-agnostic
3. Adding MySQL/PostgreSQL requires ~200 lines of code
4. LLM handles SQL dialect differences automatically
5. MongoDB stores `connectionType` to route to correct implementation
6. Full pipeline works without any other changes
