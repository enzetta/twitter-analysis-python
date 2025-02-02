from google.cloud import bigquery
import logging
import pandas as pd
from typing import List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for interacting with BigQuery."""

    def __init__(self, project_id: str, service_account_path: str):
        """Initialize BigQuery client.

        Args:
            project_id: The GCP project ID
            service_account_path: Path to service account JSON file
        """
        self.project_id = project_id
        self.client = bigquery.Client.from_service_account_json(
            service_account_path, project=project_id
        )

    def create_table(
        self, dataset_id: str, table_id: str, schema: List[bigquery.SchemaField]
    ) -> None:
        """Create a BigQuery table with retries.

        Args:
            dataset_id: The BigQuery dataset ID
            table_id: The table ID
            schema: List of BigQuery SchemaField objects defining the table schema
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        table = bigquery.Table(table_ref, schema=schema)

        # Delete existing table
        try:
            self.client.delete_table(table_ref, not_found_ok=True)
            logger.info(f"Deleted existing table: {table_ref}")
            time.sleep(5)  # Wait after deletion
        except Exception as e:
            logger.warning(f"Failed to delete existing table: {e}")

        # Create new table with retries
        max_retries = 10
        for attempt in range(max_retries):
            try:
                created_table = self.client.create_table(table)
                logger.info(f"Created table {table_ref}")

                # Verify table exists by getting it
                for _ in range(3):
                    time.sleep(10)  # Wait between checks
                    self.client.get_table(created_table.reference)

                logger.info(f"Table {table_ref} verified and ready")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Table creation attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))  # Exponential backoff

    def insert_rows_json(
        self, dataset_id: str, table_id: str, rows: List[dict]
    ) -> None:
        """Insert rows into a BigQuery table with retries.

        Args:
            dataset_id: The BigQuery dataset ID
            table_id: The table ID
            rows: List of dictionaries containing the row data
        """
        if not rows:
            return

        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        # Retry insert operation
        max_retries = 10
        for attempt in range(max_retries):
            try:
                errors = self.client.insert_rows_json(table_ref, rows)
                if errors:
                    raise Exception(f"Insert errors: {errors}")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Insert attempt {attempt + 1} failed: {e}")
                time.sleep(5 * (attempt + 1))  # Exponential backoff

    def query(self, query: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Execute a BigQuery query.

        Args:
            query: SQL query string
            limit: Optional row limit for the query

        Returns:
            DataFrame containing query results
        """
        if limit:
            query = f"{query} LIMIT {limit}"
        logger.info(f"Executing query: {query}")

        try:
            # Use legacy REST API instead of Storage API
            job_config = bigquery.QueryJobConfig(use_query_cache=True)
            query_job = self.client.query(query, job_config=job_config)
            return query_job.result().to_dataframe(create_bqstorage_client=False)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
