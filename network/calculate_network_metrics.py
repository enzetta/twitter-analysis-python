# Network metrics
import pandas as pd
import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from datetime import datetime
import logging
import os
import traceback
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

# Configuration
all_tables = {
    "all": "network_users",
    "migration": "network_migration",
    "climate": "network_climate",
}

table_name = all_tables["climate"]
dataset = "twitter_analysis_curated"
target_dataset = "python_src"  # Target dataset for storing results
project_id = "grounded-nebula-408412"

# Write behavior configuration
REPLACE_TABLE = True  # Set to False to append to existing table

# Output configuration
OUTPUT_DIR = os.path.join("data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize BigQuery client
client = bigquery.Client()

# Configure parallel backend
nx.config.backend_priority = ["parallel"]  # Use parallel backend when available
nx.config.backends.parallel.n_jobs = 2048


def fetch_monthly_data(month_start):
    """Fetch data for a specific month from BigQuery."""
    query = f"""
    SELECT *
    FROM {project_id}.{dataset}.{table_name}
    WHERE month_start = @month_start
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
    FROM {project_id}.{dataset}.{table_name}
    ORDER BY month_start
    """

    try:
        # Disable BigQuery Storage API
        job_config = bigquery.QueryJobConfig(use_query_cache=True)
        query_job = client.query(query, job_config=job_config)
        months_df = query_job.to_dataframe(create_bqstorage_client=False)
        return months_df["month_start"].tolist()
    except GoogleCloudError as e:
        logger.error(f"Error fetching months: {str(e)}")
        raise


def calculate_network_metrics(df_group, month_start):
    """Calculate network metrics for a given dataframe group."""
    try:
        # Create a directed graph from the data
        Graph = nx.from_pandas_edgelist(
            df_group,
            "source",
            "target",
            edge_attr=["weight", "sum_toxicity", "avg_toxicity"],
            create_using=nx.DiGraph(),
        )

        # Calculate network-wide toxicity metrics
        total_interactions = sum(d["weight"] for _, _, d in Graph.edges(data=True))
        total_toxicity = sum(d["sum_toxicity"] for _, _, d in Graph.edges(data=True))
        network_avg_toxicity = (
            total_toxicity / total_interactions if total_interactions > 0 else 0
        )

        # Calculate node-level toxicity metrics
        node_toxicity = {}
        for node in Graph.nodes():
            out_edges = Graph.out_edges(node, data=True)
            node_total_weight = sum(d["weight"] for _, _, d in out_edges)
            node_total_toxicity = sum(d["sum_toxicity"] for _, _, d in out_edges)
            node_toxicity[node] = (
                node_total_toxicity / node_total_weight if node_total_weight > 0 else 0
            )

        metrics = {
            "month_start": month_start,
            "table_name": table_name,
            "nodes": len(Graph),
            "edges": Graph.number_of_edges(),
            "density": nx.density(Graph),
            "connected_components": nx.number_strongly_connected_components(Graph),
            "transitivity": nx.transitivity(Graph),
            "modularity": np.nan,
            "modularity_classes": 0,
            "assortativity": np.nan,
            "network_avg_toxicity": network_avg_toxicity,
            "median_node_toxicity": (
                np.median(list(node_toxicity.values())) if node_toxicity else 0
            ),
            "max_core_number": np.nan,
            "avg_core_number": np.nan,
            "rich_club_coefficient": np.nan,
            "average_clustering": np.nan,
        }

        # Calculate metrics that require connected graphs
        if len(Graph) > 0:
            try:
                metrics["assortativity"] = nx.degree_assortativity_coefficient(Graph)

                # Create undirected version for the new metrics
                undirected_graph = Graph.to_undirected()
                undirected_graph = nx.Graph(undirected_graph)
                undirected_graph.remove_edges_from(nx.selfloop_edges(undirected_graph))

                # Calculate k-core numbers
                core_numbers = nx.core_number(undirected_graph)
                metrics["max_core_number"] = (
                    max(core_numbers.values()) if core_numbers else 0
                )
                metrics["avg_core_number"] = (
                    np.mean(list(core_numbers.values())) if core_numbers else 0
                )

                # Calculate rich club coefficient
                if metrics["max_core_number"] > 0:
                    rich_club_coeffs = nx.rich_club_coefficient(
                        undirected_graph, normalized=False
                    )
                    metrics["rich_club_coefficient"] = (
                        np.mean(list(rich_club_coeffs.values()))
                        if rich_club_coeffs
                        else 0
                    )

                # Calculate average clustering
                metrics["average_clustering"] = nx.average_clustering(undirected_graph)

            except Exception as e:
                logger.warning(
                    f"Could not calculate some metrics for timestamp {month_start}. Error: {str(e)}"
                )

            # Community detection
            try:
                logger.info(
                    f"Starting community detection for timestamp {month_start}. Graph size: {len(Graph)} nodes, {Graph.number_of_edges()} edges"
                )

                partition = community_louvain.best_partition(
                    undirected_graph, weight="weight", random_state=42
                )
                metrics["modularity"] = community_louvain.modularity(
                    partition, undirected_graph
                )
                metrics["modularity_classes"] = len(set(partition.values()))

                # Calculate average toxicity by community
                community_toxicity = {}
                for node, community_id in partition.items():
                    if community_id not in community_toxicity:
                        community_toxicity[community_id] = []
                    if node in node_toxicity:
                        community_toxicity[community_id].append(node_toxicity[node])

                community_avg_toxicity = {
                    comm: np.mean(tox) for comm, tox in community_toxicity.items()
                }

                # Add row ID for BigQuery
                metrics["row_id"] = f"{month_start.strftime('%Y-%m')}_{table_name}"

                logger.info(
                    f"Successfully completed community detection for timestamp {month_start}"
                )

            except Exception as e:
                logger.warning(
                    f"Could not calculate community metrics for timestamp {month_start}. Error: {str(e)}\n{traceback.format_exc()}"
                )

        return pd.Series(metrics)
    except Exception as e:
        logger.error(
            f"Error processing timestamp {month_start}: {str(e)}\n{traceback.format_exc()}"
        )
        return pd.Series()


def create_or_get_bigquery_table(table_name):
    """Create or get the BigQuery table for network metrics."""
    target_table = f"{project_id}.{target_dataset}.python_network_{table_name}_metrics"

    schema = [
        bigquery.SchemaField("row_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("month_start", "DATE"),
        bigquery.SchemaField("table_name", "STRING"),
        bigquery.SchemaField("nodes", "INTEGER"),
        bigquery.SchemaField("edges", "INTEGER"),
        bigquery.SchemaField("density", "FLOAT"),
        bigquery.SchemaField("connected_components", "INTEGER"),
        bigquery.SchemaField("transitivity", "FLOAT"),
        bigquery.SchemaField("modularity", "FLOAT"),
        bigquery.SchemaField("modularity_classes", "INTEGER"),
        bigquery.SchemaField("assortativity", "FLOAT"),
        bigquery.SchemaField("network_avg_toxicity", "FLOAT"),
        bigquery.SchemaField("median_node_toxicity", "FLOAT"),
        bigquery.SchemaField("max_core_number", "INTEGER"),
        bigquery.SchemaField("avg_core_number", "FLOAT"),
        bigquery.SchemaField("rich_club_coefficient", "FLOAT"),
        bigquery.SchemaField("average_clustering", "FLOAT"),
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

    return target_table


def upload_to_bigquery(df, target_table):
    """Upload network metrics to BigQuery."""
    # Convert month_start column to proper datetime type
    df["month_start"] = pd.to_datetime(df["month_start"]).dt.date

    job_config = bigquery.LoadJobConfig(
        # Use WRITE_APPEND to add to existing data
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )

    try:
        job = client.load_table_from_dataframe(df, target_table, job_config=job_config)
        job.result()
        logger.info(f"Uploaded {len(df)} rows to BigQuery")
        return True
    except Exception as e:
        logger.error(f"BigQuery upload failed: {str(e)}")
        return False


def process_network_data():
    """Process network data month by month and calculate metrics."""
    try:
        logger.info("Starting network analysis")

        # Set up BigQuery table
        target_table = create_or_get_bigquery_table(table_name)

        # Get all unique months
        months = get_all_months()
        total_months = len(months)
        logger.info(f"Found {total_months} months to process")

        metrics_list = []
        for i, month_start in enumerate(months, 1):
            logger.info(f"Processing month {i}/{total_months}: {month_start}")

            # Fetch data for this month
            df_month = fetch_monthly_data(month_start)
            logger.info(f"Fetched {len(df_month)} rows for {month_start}")

            # Calculate metrics for this month
            metrics = calculate_network_metrics(df_month, month_start)
            if not metrics.empty:
                metrics_list.append(metrics)

            logger.info(f"Completed processing for {month_start}")

        # Combine all metrics into a single DataFrame
        metrics_df = pd.DataFrame(metrics_list)

        # Save results locally with timestamp (as before)
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"{current_timestamp}_{table_name}_metrics.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        metrics_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to local file: {output_path}")

        # Upload to BigQuery
        success = upload_to_bigquery(metrics_df, target_table)
        if success:
            logger.info(f"Successfully uploaded data to BigQuery table: {target_table}")
        else:
            logger.error("Failed to upload data to BigQuery")

        return metrics_df

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
        raise


def process_table(table_key, table_val):
    """Process a single table."""
    global table_name
    try:
        logger.info(
            f"==== Starting processing for table: {table_key} ({table_val}) ===="
        )

        # Set the global table_name
        table_name = table_val

        # Process network data for this table
        metrics_df = process_network_data()

        logger.info(
            f"==== Completed processing for table: {table_key} ({table_name}) ===="
        )
        return True

    except Exception as e:
        logger.error(
            f"Processing failed for table {table_val}: {str(e)}\n{traceback.format_exc()}"
        )
        return False


def process_all_tables():
    """Process all tables sequentially."""
    try:
        logger.info(f"Starting sequential processing of {len(all_tables)} tables")

        success_count = 0
        for table_key, table_name in all_tables.items():
            if process_table(table_key, table_name):
                success_count += 1

        logger.info(
            f"Processing completed. Successfully processed {success_count}/{len(all_tables)} tables"
        )

    except Exception as e:
        logger.error(f"Overall processing failed: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    logger.info("Starting network metrics calculation")

    try:
        # Process all tables instead of just one
        process_all_tables()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
