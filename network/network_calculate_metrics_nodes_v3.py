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
    "network_users": "network_users",
    "network_hashtags": "network_hashtags",
    "network_users_interactions_only": "network_users_interactions_only",
    "network_hashtags-users": "network_hashtags-users",
}

table_name = all_tables["network_users"]
dataset = "twitter_analysis_analyses"
project_id = "grounded-nebula-408412"

LIMIT = ""  # e.g. "LIMIT 10000"
# LIMIT = "LIMIT 1000"  # e.g. "LIMIT 10000"

# Output configuration
OUTPUT_DIR = os.path.join("data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
client = bigquery.Client()

# Load politicians data
try:
    politicians_data = pd.read_csv("data/epi_data.csv")[
        ["twitter_handle", "official_name", "party"]
    ]
    logger.info(f"Loaded {len(politicians_data)} politicians from EPI data")
except Exception as e:
    df = pd.read_csv("data/epi_data.csv")
    logger.error(f"Available columns: {df.columns.tolist()}")
    logger.error(f"Error loading EPI data: {str(e)}")
    raise


def fetch_monthly_data(month_start):
    """Fetch data for a specific month from BigQuery."""
    query = f"""
    SELECT *
    FROM {project_id}.{dataset}.{table_name}
    WHERE month_start = @month_start
    ORDER BY weight DESC
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
    FROM {project_id}.{dataset}.{table_name}
    ORDER BY month_start
    """

    try:
        job_config = bigquery.QueryJobConfig(use_query_cache=True)
        query_job = client.query(query, job_config=job_config)
        months_df = query_job.to_dataframe(create_bqstorage_client=False)
        return months_df["month_start"].tolist()
    except GoogleCloudError as e:
        logger.error(f"Error fetching months: {str(e)}")
        raise


def calculate_all_metrics(df_group, month_start):
    """Calculate both network-wide and node-specific metrics."""
    try:
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

        # Network-wide metrics
        network_metrics = {
            "month_start": month_start,
            "nodes": len(Graph),
            "edges": Graph.number_of_edges(),
            "density": nx.density(Graph),
            "connected_components": nx.number_strongly_connected_components(Graph),
            "transitivity": nx.transitivity(Graph),
            "average_path_length": np.nan,
            "diameter": np.nan,
            "modularity": np.nan,
            "modularity_classes": 0,
            "assortativity": np.nan,
            "network_avg_toxicity": network_avg_toxicity,
            "max_node_toxicity": max(node_toxicity.values()) if node_toxicity else 0,
            "min_node_toxicity": min(node_toxicity.values()) if node_toxicity else 0,
            "median_node_toxicity": (
                np.median(list(node_toxicity.values())) if node_toxicity else 0
            ),
        }

        # Calculate metrics that require connected graphs
        if len(Graph) > 0:
            try:
                network_metrics["assortativity"] = nx.degree_assortativity_coefficient(
                    Graph
                )

                if nx.is_weakly_connected(Graph):
                    lengths = dict(
                        nx.all_pairs_dijkstra_path_length(Graph, weight="weight")
                    )
                    all_lengths = [
                        length
                        for target_lengths in lengths.values()
                        for length in target_lengths.values()
                    ]
                    if all_lengths:
                        network_metrics["average_path_length"] = np.mean(all_lengths)
                        network_metrics["diameter"] = max(all_lengths)

                # Community detection
                undirected_graph = Graph.to_undirected()
                undirected_graph = nx.Graph(undirected_graph)
                undirected_graph.remove_edges_from(nx.selfloop_edges(undirected_graph))

                partition = community_louvain.best_partition(
                    undirected_graph, weight="weight", random_state=42
                )
                network_metrics["modularity"] = community_louvain.modularity(
                    partition, undirected_graph
                )
                network_metrics["modularity_classes"] = len(set(partition.values()))

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

                network_metrics["max_community_toxicity"] = (
                    max(community_avg_toxicity.values())
                    if community_avg_toxicity
                    else 0
                )
                network_metrics["min_community_toxicity"] = (
                    min(community_avg_toxicity.values())
                    if community_avg_toxicity
                    else 0
                )

            except Exception as e:
                logger.warning(f"Some metrics calculation failed: {str(e)}")

        # Node-specific metrics
        node_metrics = []
        if len(Graph) > 0:
            logger.info("calculating: pagerank")
            pagerank = nx.pagerank(Graph, weight="weight")

            # Get nodes sorted by PageRank first
            sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            top_pagerank_nodes = {node for node, _ in sorted_nodes[:1000]}

            # Combine politician handles and top PageRank nodes
            nodes_of_interest = (
                set(politicians_data["twitter_handle"]) | top_pagerank_nodes
            )

            logger.info("calculating: in_degree")
            in_degree = dict(Graph.in_degree(weight="weight"))
            logger.info("calculating: out_degree")
            out_degree = dict(Graph.out_degree(weight="weight"))

            logger.info("calculating: betweenness")
            k = min(1000, max(50, int(len(Graph) * 0.02)))
            betweenness = nx.betweenness_centrality(Graph, weight="weight", k=k)

            logger.info("calculating: clustering")
            clustering = nx.clustering(Graph.to_undirected())

            # Create set of politician handles for faster lookup
            politician_handles = set(politicians_data["twitter_handle"])

            # Track processed nodes to avoid duplicates
            processed_nodes = set()

            # Process politicians
            for _, politician in politicians_data.iterrows():
                handle = politician["twitter_handle"]
                if handle not in Graph.nodes():
                    continue

                processed_nodes.add(handle)

                out_edges = Graph.out_edges(handle, data=True)
                in_edges = Graph.in_edges(handle, data=True)
                sent_toxicity = sum(d["sum_toxicity"] for _, _, d in out_edges)
                received_toxicity = sum(d["sum_toxicity"] for _, _, d in in_edges)
                sent_interactions = sum(d["weight"] for _, _, d in out_edges)
                received_interactions = sum(d["weight"] for _, _, d in in_edges)

                node_metrics.append(
                    {
                        "month_start": month_start,
                        "twitter_handle": handle,
                        "official_name": politician["official_name"],
                        "party": politician["party"],
                        "category": "politician",
                        "pagerank": pagerank.get(handle, 0),
                        # "closeness_centrality": closeness_centrality.get(handle, 0),
                        "in_degree": in_degree.get(handle, 0),
                        "out_degree": out_degree.get(handle, 0),
                        "betweenness": betweenness.get(handle, 0),
                        "clustering_coefficient": clustering.get(handle, 0),
                        "sent_toxicity": sent_toxicity,
                        "received_toxicity": received_toxicity,
                        "avg_sent_toxicity": (
                            sent_toxicity / sent_interactions
                            if sent_interactions > 0
                            else 0
                        ),
                        "avg_received_toxicity": (
                            received_toxicity / received_interactions
                            if received_interactions > 0
                            else 0
                        ),
                        "total_interactions": sent_interactions + received_interactions,
                    }
                )

            # Process top 1000 accounts by PageRank that aren't politicians
            top_accounts_added = 0
            for node, rank in sorted_nodes:
                if top_accounts_added >= 1000:
                    break

                if node in processed_nodes:
                    continue

                out_edges = Graph.out_edges(node, data=True)
                in_edges = Graph.in_edges(node, data=True)
                sent_toxicity = sum(d["sum_toxicity"] for _, _, d in out_edges)
                received_toxicity = sum(d["sum_toxicity"] for _, _, d in in_edges)
                sent_interactions = sum(d["weight"] for _, _, d in out_edges)
                received_interactions = sum(d["weight"] for _, _, d in in_edges)

                node_metrics.append(
                    {
                        "month_start": month_start,
                        "twitter_handle": node,
                        "official_name": None,
                        "party": None,
                        "category": "top_pagerank",
                        "pagerank": pagerank.get(node, 0),
                        # "closeness_centrality": closeness_centrality.get(node, 0),
                        "in_degree": in_degree.get(node, 0),
                        "out_degree": out_degree.get(node, 0),
                        "betweenness": betweenness.get(node, 0),
                        "clustering_coefficient": clustering.get(node, 0),
                        "sent_toxicity": sent_toxicity,
                        "received_toxicity": received_toxicity,
                        "avg_sent_toxicity": (
                            sent_toxicity / sent_interactions
                            if sent_interactions > 0
                            else 0
                        ),
                        "avg_received_toxicity": (
                            received_toxicity / received_interactions
                            if received_interactions > 0
                            else 0
                        ),
                        "total_interactions": sent_interactions + received_interactions,
                    }
                )
                top_accounts_added += 1

        return pd.Series(network_metrics), pd.DataFrame(node_metrics)

    except Exception as e:
        logger.error(f"Error processing metrics: {str(e)}\n{traceback.format_exc()}")
        return pd.Series(), pd.DataFrame()


def process_all_data():
    """Process all network data and calculate both sets of metrics."""
    try:
        months = get_all_months()
        total_months = len(months)
        logger.info(f"Found {total_months} months to process")

        network_metrics_list = []
        all_node_metrics = []

        # Track processing times
        start_time = datetime.now()
        iteration_times = []

        for i, month_start in enumerate(months, 1):
            iteration_start = datetime.now()
            logger.info(f"Processing month {i}/{total_months}: {month_start}")

            df_month = fetch_monthly_data(month_start)
            logger.info(f"Fetched {len(df_month)} rows for {month_start}")

            network_metrics, node_metrics = calculate_all_metrics(df_month, month_start)

            if not network_metrics.empty:
                network_metrics_list.append(network_metrics)
            if not node_metrics.empty:
                all_node_metrics.append(node_metrics)

            # Calculate timing metrics
            iteration_end = datetime.now()
            iteration_duration = iteration_end - iteration_start

            logger.info(f"Completed processing for {month_start}")
            logger.info(f"Iteration duration: {iteration_duration}")
            logger.info(f"Iteration end: {iteration_end}")

        # Calculate total processing time
        total_duration = datetime.now() - start_time
        logger.info(f"Total processing time: {total_duration}")

        # Rest of the function remains the same...
        network_df = pd.DataFrame(network_metrics_list)
        node_df = pd.concat(all_node_metrics, ignore_index=True)

        party_df = (
            node_df[node_df["category"] == "politician"]
            .groupby(["month_start", "party"])
            .agg(
                {
                    "pagerank": ["mean", "std"],
                    "in_degree": ["mean", "std"],
                    "out_degree": ["mean", "std"],
                    "betweenness": ["mean", "std"],
                    "clustering_coefficient": ["mean", "std"],
                    "sent_toxicity": "sum",
                    "received_toxicity": "sum",
                    "total_interactions": "sum",
                }
            )
            .reset_index()
        )

        party_df.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in party_df.columns
        ]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        network_df.to_csv(
            os.path.join(OUTPUT_DIR, f"network_metrics_{timestamp}.csv"), index=False
        )
        node_df.to_csv(
            os.path.join(OUTPUT_DIR, f"node_metrics_{timestamp}.csv"), index=False
        )
        party_df.to_csv(
            os.path.join(OUTPUT_DIR, f"party_metrics_{timestamp}.csv"), index=False
        )

        logger.info(f"Saved network metrics to: network_metrics_{timestamp}.csv")
        logger.info(f"Saved node metrics to: node_metrics_{timestamp}.csv")
        logger.info(f"Saved party metrics to: party_metrics_{timestamp}.csv")

        return network_df, node_df, party_df

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        network_metrics, node_metrics, party_metrics = process_all_data()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
