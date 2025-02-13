import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
import logging
import traceback
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

# Configuration
project_id = "grounded-nebula-408412"
dataset = "twitter_analysis_curated"
source_table = "network_users"
target_table = f"{project_id}.{dataset}.python_node_metrics_monthly"

# Write behavior configuration
REPLACE_TABLE = True  # Set to False to append to existing table
LIMIT = f"""
ORDER BY weight DESC
"""
# LIMIT 50000

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
client = bigquery.Client()

# Configure parallel backend
nx.config.backend_priority = ["parallel"]  # Use parallel backend when available
nx.config.backends.parallel.n_jobs = 2048


def fetch_monthly_data(month_start):
    """Fetch data for a specific month from BigQuery."""
    query = f"""
    SELECT *
    FROM {project_id}.{dataset}.{source_table}
    WHERE month_start = @month_start
    {LIMIT}
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("month_start", "TIMESTAMP", month_start)
        ],
        use_query_cache=True,
    )

    try:
        query_job = client.query(query, job_config=job_config)
        df = query_job.to_dataframe(create_bqstorage_client=False)
        return df
    except GoogleCloudError as e:
        logger.error(f"BigQuery error for {month_start}: {str(e)}")
        raise


def get_all_months():
    """Get list of all unique months in the dataset."""
    query = f"""
    SELECT DISTINCT month_start
    FROM {project_id}.{dataset}.{source_table}
    ORDER BY month_start
    """

    try:
        query_job = client.query(query)
        months_df = query_job.to_dataframe(create_bqstorage_client=False)
        return months_df["month_start"].tolist()
    except GoogleCloudError as e:
        logger.error(f"Error fetching months: {str(e)}")
        raise


def calculate_node_metrics(df_month, month_start):
    """Calculate node-level metrics for the network."""
    try:
        # Create directed graph
        Graph = nx.from_pandas_edgelist(
            df_month,
            "source",
            "target",
            edge_attr=["weight", "sum_toxicity", "avg_toxicity"],
            create_using=nx.DiGraph(),
        )

        if len(Graph) == 0:
            logger.warning(f"Empty graph for {month_start}")
            return pd.DataFrame()

        logger.info(f"Calculating metrics for {len(Graph)} nodes")

        # Calculate all centrality metrics
        logger.info("Calculating PageRank")
        pagerank = nx.pagerank(Graph, weight="weight")

        logger.info("Calculating degree")
        degree_in = dict(Graph.in_degree(weight="weight"))
        degree_out = dict(Graph.out_degree(weight="weight"))

        logger.info("Calculating betweenness")
        k = min(1000, max(50, int(len(Graph) * 0.02)))  # Sampling for large graphs
        betweenness = nx.betweenness_centrality(Graph, weight="weight", k=k)

        # logger.info("Calculating closeness centrality")
        # closeness = nx.closeness_centrality(Graph, distance="weight")

        logger.info("Calculating clustering coefficients and triangles")
        undirected = Graph.to_undirected()
        clustering = nx.clustering(undirected)
        triangles = nx.triangles(undirected)

        logger.info("Calculating k-core numbers")
        core_numbers = nx.core_number(undirected)

        # Calculate node-specific metrics
        node_metrics = []
        for node in Graph.nodes():
            out_edges = Graph.out_edges(node, data=True)
            in_edges = Graph.in_edges(node, data=True)

            # Calculate interaction and toxicity metrics
            interactions_sent = sum(d["weight"] for _, _, d in out_edges)
            interactions_received = sum(d["weight"] for _, _, d in in_edges)
            toxicity_sent = sum(d["sum_toxicity"] for _, _, d in out_edges)
            toxicity_received = sum(d["sum_toxicity"] for _, _, d in in_edges)

            # Create unique row identifier
            row_id = f"{month_start.strftime('%Y-%m')}_{node}"

            # Compile metrics
            node_metrics.append(
                {
                    "row_id": row_id,
                    "month_start": month_start,
                    "node_id": node,
                    "pagerank": pagerank.get(node, 0),
                    "degree_in": degree_in.get(node, 0),
                    "degree_out": degree_out.get(node, 0),
                    "betweenness": betweenness.get(node, 0),
                    # "closeness": closeness.get(node, 0),
                    "clustering": clustering.get(node, 0),
                    "triangles": triangles.get(node, 0),
                    "core_number": core_numbers.get(node, 0),
                    "interactions_sent": interactions_sent,
                    "interactions_received": interactions_received,
                    "interactions_total": interactions_sent + interactions_received,
                    "toxicity_sent": toxicity_sent,
                    "toxicity_received": toxicity_received,
                    "toxicity_sent_avg": (
                        toxicity_sent / interactions_sent
                        if interactions_sent > 0
                        else 0
                    ),
                    "toxicity_received_avg": (
                        toxicity_received / interactions_received
                        if interactions_received > 0
                        else 0
                    ),
                }
            )

        return pd.DataFrame(node_metrics)

    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}\n{traceback.format_exc()}")
        return pd.DataFrame()


def create_or_get_table():
    """Create the BigQuery table if it doesn't exist or get the existing schema."""
    schema = [
        bigquery.SchemaField("row_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("month_start", "TIMESTAMP"),
        bigquery.SchemaField("node_id", "STRING"),
        bigquery.SchemaField("pagerank", "FLOAT"),
        bigquery.SchemaField("degree_in", "FLOAT"),
        bigquery.SchemaField("degree_out", "FLOAT"),
        bigquery.SchemaField("betweenness", "FLOAT"),
        # bigquery.SchemaField("closeness", "FLOAT"),
        bigquery.SchemaField("clustering", "FLOAT"),
        bigquery.SchemaField("triangles", "INTEGER"),
        bigquery.SchemaField("core_number", "INTEGER"),
        bigquery.SchemaField("interactions_sent", "INTEGER"),
        bigquery.SchemaField("interactions_received", "INTEGER"),
        bigquery.SchemaField("interactions_total", "INTEGER"),
        bigquery.SchemaField("toxicity_sent", "FLOAT"),
        bigquery.SchemaField("toxicity_received", "FLOAT"),
        bigquery.SchemaField("toxicity_sent_avg", "FLOAT"),
        bigquery.SchemaField("toxicity_received_avg", "FLOAT"),
    ]

    table = bigquery.Table(target_table, schema=schema)

    if REPLACE_TABLE:
        # Delete the table if it exists and create a new one
        try:
            client.delete_table(table)
            logger.info(f"Deleted existing table {target_table}")
        except Exception:
            pass  # Table might not exist

        table.clustering_fields = ["month_start"]
        table = client.create_table(table)
        logger.info(f"Created new table {target_table}")
    else:
        try:
            # Get existing table
            table = client.get_table(target_table)
            logger.info(f"Using existing table {target_table}")
        except Exception:
            # Create new table if it doesn't exist
            table.clustering_fields = ["month_start"]
            table = client.create_table(table)
            logger.info(f"Created new table {target_table}")

    return table


def upload_to_bigquery(df, month_start):
    """Upload node metrics to BigQuery."""
    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if REPLACE_TABLE
            else bigquery.WriteDisposition.WRITE_APPEND
        ),
    )

    try:
        job = client.load_table_from_dataframe(df, target_table, job_config=job_config)
        job.result()
        logger.info(f"Uploaded {len(df)} rows for {month_start}")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise


def process_all_months():
    """Process all months and calculate node metrics."""
    try:
        # Create or get the table first
        table = create_or_get_table()

        months = get_all_months()
        total_months = len(months)
        logger.info(f"Processing {total_months} months")

        for i, month_start in enumerate(months, 1):
            logger.info(f"Processing month {i}/{total_months}: {month_start}")

            # Get data for the month
            df_month = fetch_monthly_data(month_start)
            logger.info(f"Fetched {len(df_month)} rows")

            # Calculate metrics
            node_metrics = calculate_node_metrics(df_month, month_start)

            if not node_metrics.empty:
                # Upload to BigQuery
                upload_to_bigquery(node_metrics, month_start)
            else:
                logger.warning(f"No metrics calculated for {month_start}")

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        process_all_months()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
