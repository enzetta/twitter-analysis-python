import pandas as pd
import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from datetime import datetime
import logging
import os
import traceback

file_name = "hashtags-users.csv"
input_path = os.path.join("data", "input", file_name)
output_path = os.path.join("data", "output", file_name)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
                metrics["assortativity"] = nx.degree_assortativity_coefficient(Graph)
            except Exception as e:
                logger.warning(
                    f"Could not calculate assortativity for timestamp {month_start}. Error: {str(e)}"
                )
                metrics["assortativity"] = np.nan

            # Path length calculations for weakly connected graphs
            if nx.is_weakly_connected(Graph):
                try:
                    lengths = dict(
                        nx.all_pairs_dijkstra_path_length(Graph, weight="weight")
                    )
                    all_lengths = [
                        length
                        for target_lengths in lengths.values()
                        for length in target_lengths.values()
                    ]
                    if all_lengths:
                        metrics["average_path_length"] = np.mean(all_lengths)
                        metrics["diameter"] = max(all_lengths)
                except Exception as e:
                    logger.warning(
                        f"Could not calculate path metrics for timestamp {month_start}. Error: {str(e)}"
                    )

            # Community detection
            try:
                logger.info(
                    f"Starting community detection for timestamp {month_start}. Graph size: {len(Graph)} nodes, {Graph.number_of_edges()} edges"
                )

                undirected_graph = Graph.to_undirected()
                undirected_graph = nx.Graph(undirected_graph)
                undirected_graph.remove_edges_from(nx.selfloop_edges(undirected_graph))

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

                metrics["max_community_toxicity"] = (
                    max(community_avg_toxicity.values())
                    if community_avg_toxicity
                    else 0
                )
                metrics["min_community_toxicity"] = (
                    min(community_avg_toxicity.values())
                    if community_avg_toxicity
                    else 0
                )

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


def process_network_data(input_path):
    """Process network data and calculate metrics for each timestamp."""
    try:
        logger.info(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        df["month_start"] = pd.to_datetime(df["month_start"])

        total_rows = len(df)
        logger.info(f"Total rows to process: {total_rows}")

        metrics_list = []
        processed_rows = 0

        for month_start, group in df.groupby("month_start"):
            logger.info(f"Processing timestamp {month_start} with {len(group)} rows")
            metrics = calculate_network_metrics(group, month_start)
            if not metrics.empty:
                metrics_list.append(metrics)

            processed_rows += len(group)
            if processed_rows % 10000 == 0:
                logger.info(
                    f"Processed {processed_rows}/{total_rows} rows ({(processed_rows/total_rows)*100:.2f}%)"
                )

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        return metrics_df

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    logger.info("Starting network metrics calculation")

    try:
        metrics_df = process_network_data(input_path)
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
