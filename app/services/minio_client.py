"""
MinIO client for storing query results as Parquet files.
"""
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import io
import pandas as pd
from minio import Minio
from minio.error import S3Error

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("minio_client")


class MinIOClient:
    """MinIO client for object storage."""

    def __init__(self):
        """Initialize MinIO client."""
        self.client = None
        self.bucket = settings.MINIO_BUCKET
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to MinIO server.

        Returns:
            True if connection successful
        """
        try:
            self.client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE
            )

            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created MinIO bucket: {self.bucket}")

            self.connected = True
            logger.info(f"Connected to MinIO: {settings.MINIO_ENDPOINT}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            self.connected = False
            return False

    def upload_parquet(
        self,
        data: List[Dict[str, Any]],
        query: str,
        database_id: str
    ) -> str:
        """
        Upload query results as Parquet file to MinIO.

        Args:
            data: Query results as list of dictionaries
            query: Original natural language query
            database_id: Database ID

        Returns:
            MinIO object path
        """
        if not self.connected:
            raise ConnectionError("MinIO not connected")

        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{database_id}_{timestamp}.parquet"
            object_path = f"query-results/{filename}"

            # Convert DataFrame to Parquet bytes
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, engine='pyarrow', index=False)
            parquet_bytes = parquet_buffer.getvalue()
            parquet_size = len(parquet_bytes)

            # Upload to MinIO
            parquet_buffer.seek(0)
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=object_path,
                data=parquet_buffer,
                length=parquet_size,
                content_type="application/octet-stream"
            )

            logger.info(f"Uploaded Parquet file to MinIO: {object_path} ({parquet_size} bytes)")
            return object_path

        except Exception as e:
            logger.error(f"Error uploading Parquet file: {e}")
            raise

    def download_parquet(self, object_path: str) -> List[Dict[str, Any]]:
        """
        Download and read Parquet file from MinIO.

        Args:
            object_path: MinIO object path

        Returns:
            List of dictionaries
        """
        if not self.connected:
            raise ConnectionError("MinIO not connected")

        try:
            # Download from MinIO
            response = self.client.get_object(self.bucket, object_path)
            parquet_data = response.read()
            response.close()
            response.release_conn()

            # Read Parquet
            df = pd.read_parquet(io.BytesIO(parquet_data))
            return df.to_dict(orient='records')

        except S3Error as e:
            logger.error(f"Error downloading Parquet file: {e}")
            raise

    def get_presigned_url(self, object_path: str, expiry_seconds: int = 3600) -> str:
        """
        Get presigned URL for downloading the file.

        Args:
            object_path: MinIO object path
            expiry_seconds: URL expiry time in seconds (default: 1 hour)

        Returns:
            Presigned URL
        """
        if not self.connected:
            raise ConnectionError("MinIO not connected")

        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket_name=self.bucket,
                object_name=object_path,
                expires=timedelta(seconds=expiry_seconds)
            )
            return url

        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise

    def disconnect(self):
        """Disconnect from MinIO."""
        if self.connected:
            self.client = None
            self.connected = False
            logger.info("Disconnected from MinIO")


# Global singleton instance
_minio_client = None


def get_minio_client() -> MinIOClient:
    """Get MinIO client singleton."""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinIOClient()
        _minio_client.connect()
    return _minio_client
