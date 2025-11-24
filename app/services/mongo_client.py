"""
MongoDB client for fetching table metadata.
"""
from typing import List, Dict, Any
from pymongo import MongoClient
from bson import ObjectId
from app.config import settings
from app.utils.logger import setup_logger
from app.services.schema_extractor import SchemaExtractor

logger = setup_logger("mongo_client")


class MongoDBClient:
    """MongoDB client for table metadata."""

    def __init__(self):
        """Initialize MongoDB client."""
        self.client = None
        self.db = None
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to MongoDB.

        Returns:
            True if connection successful
        """
        try:
            self.client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            self.db = self.client[settings.MONGO_DB_NAME]
            self.connected = True
            logger.info(f"Connected to MongoDB: {settings.MONGO_DB_NAME}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")

    def fetch_tables_for_database(self, db_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all tables for a given database ID by connecting directly to the database.

        Args:
            db_id: MongoDB database document ID

        Returns:
            List of table metadata dicts with structure:
            [
                {
                    "table_name": str,
                    "name": str,
                    "description": str,
                    "columns": [{"name": str, "type": str, ...}],
                    "category": str,
                    "row_count": int
                },
                ...
            ]
        """
        if not self.connected:
            raise ConnectionError("MongoDB not connected")

        try:
            # Fetch database document
            databases_collection = self.db[settings.MONGO_COLLECTION]
            db_doc = databases_collection.find_one({"_id": ObjectId(db_id)})

            if not db_doc:
                logger.error(f"Database with ID {db_id} not found")
                return []

            # Get connection ID and database name
            connection_id = db_doc.get("connectionId")
            database_name = db_doc.get("databaseName")

            if not connection_id or not database_name:
                logger.error(f"Database document missing connectionId or databaseName")
                return []

            logger.info(f"Fetching tables for database '{database_name}' using connection '{connection_id}'")

            # Fetch connection details from connections collection
            connections_collection = self.db["connections"]
            connection_doc = connections_collection.find_one({"_id": ObjectId(connection_id)})

            if not connection_doc:
                logger.error(f"Connection with ID {connection_id} not found")
                return []

            # Prepare connection config for schema extractor
            connection_config = {
                "connectionType": connection_doc.get("connectionType", "mssql"),
                "host": connection_doc.get("host"),
                "port": connection_doc.get("port", 1433),
                "username": connection_doc.get("username"),
                "password": connection_doc.get("password"),  # Encrypted
            }

            # Extract schemas using SchemaExtractor (streaming generator)
            logger.info(f"Connecting to {connection_config['connectionType']} database to extract schemas")
            extractor = SchemaExtractor(connection_config)

            # Return generator for streaming processing
            return extractor.extract_schemas(database_name)

        except Exception as e:
            logger.error(f"Error fetching tables: {e}")
            raise

    def get_table_metadata(self, db_id: str, table_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific table.

        Args:
            db_id: Database ID
            table_name: Table name

        Returns:
            Table metadata dict with columns info
        """
        # Fetch all tables for the database (this should be cached by the caller)
        tables = self.fetch_tables_for_database(db_id)

        # Find the specific table
        for table in tables:
            if table.get('table_name') == table_name or table.get('name') == table_name:
                return table

        logger.warning(f"Table '{table_name}' not found in database {db_id}")
        return None

    def get_database_connection_config(self, db_id: str) -> Dict[str, Any]:
        """
        Get connection configuration for a database.

        Args:
            db_id: MongoDB database document ID

        Returns:
            Dict with connection config and database name:
            {
                "database_name": str,
                "connection_config": {
                    "connectionType": str,
                    "host": str,
                    "port": int,
                    "username": str,
                    "password": str
                }
            }
        """
        if not self.connected:
            raise ConnectionError("MongoDB not connected")

        try:
            # Fetch database document
            databases_collection = self.db[settings.MONGO_COLLECTION]
            db_doc = databases_collection.find_one({"_id": ObjectId(db_id)})

            if not db_doc:
                raise ValueError(f"Database with ID {db_id} not found")

            # Get connection ID and database name
            connection_id = db_doc.get("connectionId")
            database_name = db_doc.get("databaseName")

            if not connection_id or not database_name:
                raise ValueError(f"Database document missing connectionId or databaseName")

            # Fetch connection details
            connections_collection = self.db["connections"]
            connection_doc = connections_collection.find_one({"_id": ObjectId(connection_id)})

            if not connection_doc:
                raise ValueError(f"Connection with ID {connection_id} not found")

            # Prepare connection config
            connection_config = {
                "connectionType": connection_doc.get("connectionType", "mssql"),
                "host": connection_doc.get("host"),
                "port": connection_doc.get("port", 1433),
                "username": connection_doc.get("username"),
                "password": connection_doc.get("password"),  # Encrypted
            }

            return {
                "database_name": database_name,
                "connection_config": connection_config
            }

        except Exception as e:
            logger.error(f"Error getting connection config: {e}")
            raise


# Global singleton instance
_mongo_client = None


def get_mongo_client() -> MongoDBClient:
    """Get MongoDB client singleton."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoDBClient()
        _mongo_client.connect()
    return _mongo_client
