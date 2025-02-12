import pandas as pd
import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from datetime import datetime
import logging
import os
import traceback
import json
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
import scipy.stats

# Configuration
all_tables = {
    "network_users": "network_users",
    "network_hashtags": "network_hashtags",
    "network_users_interactions_only": "network_users_interactions_only",
    "network_hashtags-users": "network_hashtags-users",
}

table_name = all_tables["network_users"]
dataset = "twitter_analysis_curated"
project_id = "grounded-nebula-408412"
# LIMIT = ""
LIMIT = "LIMIT 100000"

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


def fetch_monthly_data(month_start):
    """Fetch data for a specific month from BigQuery."""
    query = f"""
    SELECT *
    FROM {project_id}.{dataset}.{table_name}
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


def calculate_advanced_metrics(Graph):
    """Calculate additional advanced network metrics."""
    metrics = {}

    try:
        # 1. Assortativity (do similar nodes connect to each other?)
        logger.info("Calculating Assortativity")
        metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(Graph)

        # 2. Reciprocity (proportion of mutual connections)
        logger.info("Calculating Reciprocity")
        metrics["reciprocity"] = nx.reciprocity(Graph)

        # 3. Core numbers (k-core decomposition)
        logger.info("Calculating k-core decomposition")
        core_numbers = nx.core_number(Graph.to_undirected())
        metrics["max_core_number"] = max(core_numbers.values())
        metrics["avg_core_number"] = np.mean(list(core_numbers.values()))

        # 4. Network resilience metrics
        logger.info("Calculating Network resilience metrics")
        # Sample a subset of nodes for efficiency
        n_samples = min(1000, max(50, int(len(Graph) * 0.02)))
        sampled_nodes = np.random.choice(
            list(Graph.nodes()), size=n_samples, replace=False
        )

        # Calculate impact of node removals
        impacts = []
        largest_comp_size = len(max(nx.weakly_connected_components(Graph), key=len))

        for node in sampled_nodes:
            G_temp = Graph.copy()
            G_temp.remove_node(node)
            new_size = len(max(nx.weakly_connected_components(G_temp), key=len))
            impact = (largest_comp_size - new_size) / largest_comp_size
            impacts.append(impact)

        metrics["avg_node_removal_impact"] = np.mean(impacts)
        metrics["max_node_removal_impact"] = max(impacts)

        # 5. Rich club coefficient (are high-degree nodes well connected?)
        # Calculate for top 10% of nodes by degree
        logger.info("Calculating Rich club coefficient")
        degrees = dict(Graph.degree())
        threshold = np.percentile(list(degrees.values()), 90)
        rich_nodes = [n for n, d in degrees.items() if d >= threshold]
        rich_subgraph = Graph.subgraph(rich_nodes)
        metrics["rich_club_density"] = nx.density(rich_subgraph)

        # 6. Flow metrics
        # Sample pairs of nodes for current flow analysis
        logger.info("Calculating Flow metrics")
        n_flow_samples = min(100, len(Graph))
        flow_pairs = np.random.choice(list(Graph.nodes()), size=(n_flow_samples, 2))

        current_flows = []
        for source, target in flow_pairs:
            try:
                flow = nx.edge_current_flow_betweenness_centrality(
                    Graph.to_undirected(), normalized=True, solver="full"
                )
                current_flows.append(np.mean(list(flow.values())))
            except:
                continue

        if current_flows:
            metrics["avg_current_flow"] = np.mean(current_flows)

        # 7. Hierarchy measures
        # Calculate fraction of edges participating in cycles
        logger.info("Calculating Hierarchy measures")
        try:
            cycles = list(nx.simple_cycles(Graph))
            cycle_edges = set()
            for cycle in cycles:
                for i in range(len(cycle)):
                    cycle_edges.add((cycle[i], cycle[(i + 1) % len(cycle)]))
            metrics["cycle_edge_ratio"] = len(cycle_edges) / Graph.number_of_edges()
        except:
            metrics["cycle_edge_ratio"] = None

        # 8. Modularity (using existing community detection)
        logger.info("Calculating Modularity")
        try:
            partition = community_louvain.best_partition(Graph.to_undirected())
            metrics["modularity"] = community_louvain.modularity(
                partition, Graph.to_undirected()
            )
        except:
            metrics["modularity"] = None

        return metrics

    except Exception as e:
        logger.error(f"Error calculating advanced metrics: {str(e)}")
        return None


def save_advanced_metrics(Graph, output_folder):
    """Calculate and save advanced network metrics."""
    try:
        advanced_metrics = calculate_advanced_metrics(Graph)
        if advanced_metrics:
            # Convert to DataFrame for saving
            metrics_df = pd.DataFrame([advanced_metrics])
            metrics_df.to_csv(
                os.path.join(output_folder, "advanced_metrics.csv"), index=False
            )

            # Also save as JSON for easier reading
            with open(os.path.join(output_folder, "advanced_metrics.json"), "w") as f:
                json.dump(advanced_metrics, f, indent=2)

        return advanced_metrics
    except Exception as e:
        logger.error(f"Error saving advanced metrics: {str(e)}")
        return None


def calculate_histogram_data(Graph, month_start, n_bins=50):
    """Calculate histogram data for key network metrics."""
    try:
        # Convert Timestamp to string for JSON serialization
        histogram_data = {
            "month_start": month_start.strftime("%Y-%m-%d %H:%M:%S%z"),
            "histograms": {},
        }

        if len(Graph) > 0:
            # 1. Degree Distributions
            in_degrees = [d for _, d in Graph.in_degree()]
            out_degrees = [d for _, d in Graph.out_degree()]
            total_degrees = [d for _, d in Graph.degree()]

            # Calculate histograms with both linear and log bins
            for scale in ["linear", "log"]:
                if scale == "log":
                    # For log scale, filter out zeros and take log
                    in_deg_nonzero = [d for d in in_degrees if d > 0]
                    out_deg_nonzero = [d for d in out_degrees if d > 0]
                    total_deg_nonzero = [d for d in total_degrees if d > 0]

                    if in_deg_nonzero:
                        bins = np.logspace(
                            np.log10(min(in_deg_nonzero)),
                            np.log10(max(in_deg_nonzero)),
                            n_bins,
                        )
                        hist, bin_edges = np.histogram(in_deg_nonzero, bins=bins)
                        histogram_data["histograms"][f"in_degree_hist_log"] = {
                            "counts": hist.tolist(),
                            "bin_edges": bin_edges.tolist(),
                        }

                    if out_deg_nonzero:
                        bins = np.logspace(
                            np.log10(min(out_deg_nonzero)),
                            np.log10(max(out_deg_nonzero)),
                            n_bins,
                        )
                        hist, bin_edges = np.histogram(out_deg_nonzero, bins=bins)
                        histogram_data["histograms"][f"out_degree_hist_log"] = {
                            "counts": hist.tolist(),
                            "bin_edges": bin_edges.tolist(),
                        }

                    if total_deg_nonzero:
                        bins = np.logspace(
                            np.log10(min(total_deg_nonzero)),
                            np.log10(max(total_deg_nonzero)),
                            n_bins,
                        )
                        hist, bin_edges = np.histogram(total_deg_nonzero, bins=bins)
                        histogram_data["histograms"][f"total_degree_hist_log"] = {
                            "counts": hist.tolist(),
                            "bin_edges": bin_edges.tolist(),
                        }
                else:
                    # Linear scale histograms
                    hist, bin_edges = np.histogram(in_degrees, bins=n_bins)
                    histogram_data["histograms"]["in_degree_hist"] = {
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                    }

                    hist, bin_edges = np.histogram(out_degrees, bins=n_bins)
                    histogram_data["histograms"]["out_degree_hist"] = {
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                    }

                    hist, bin_edges = np.histogram(total_degrees, bins=n_bins)
                    histogram_data["histograms"]["total_degree_hist"] = {
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                    }

            # 2. Edge Weight Distribution (both linear and log scales)
            edge_weights = [d["weight"] for _, _, d in Graph.edges(data=True)]
            # Linear scale
            hist, bin_edges = np.histogram(edge_weights, bins=n_bins)
            histogram_data["histograms"]["edge_weight_hist"] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
            }
            # Log scale
            edge_weights_nonzero = [w for w in edge_weights if w > 0]
            if edge_weights_nonzero:
                bins = np.logspace(
                    np.log10(min(edge_weights_nonzero)),
                    np.log10(max(edge_weights_nonzero)),
                    n_bins,
                )
                hist, bin_edges = np.histogram(edge_weights_nonzero, bins=bins)
                histogram_data["histograms"]["edge_weight_hist_log"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }

            # 3. Clustering Coefficient Distribution
            try:
                clustering_coeffs = list(nx.clustering(Graph).values())
                hist, bin_edges = np.histogram(clustering_coeffs, bins=n_bins)
                histogram_data["histograms"]["clustering_hist"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            except Exception as e:
                logger.warning(f"Could not calculate clustering histogram: {str(e)}")

            # 4. Path Length Distribution (for largest component)
            try:
                largest_component = max(nx.weakly_connected_components(Graph), key=len)
                largest_subgraph = Graph.subgraph(largest_component)

                path_lengths = []
                # Sample nodes for efficiency if graph is large
                sample_size = min(1000, max(50, int(len(Graph) * 0.02)))
                sampled_nodes = np.random.choice(
                    list(largest_subgraph.nodes()), size=sample_size, replace=False
                )

                for node in sampled_nodes:
                    lengths = nx.single_source_shortest_path_length(
                        largest_subgraph, node
                    )
                    path_lengths.extend(lengths.values())

                hist, bin_edges = np.histogram(path_lengths, bins=n_bins)
                histogram_data["histograms"]["path_length_hist"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            except Exception as e:
                logger.warning(f"Could not calculate path length histogram: {str(e)}")

        return histogram_data

    except Exception as e:
        logger.error(f"Error calculating histogram data: {str(e)}")
        return None


def save_data_tables(month_start, Graph, output_folder):
    """Save various network analysis tables to CSV files."""
    try:
        # 1. Degree Data
        degree_data = pd.DataFrame(
            {
                "node": list(Graph.nodes()),
                "in_degree": [d for _, d in Graph.in_degree()],
                "out_degree": [d for _, d in Graph.out_degree()],
                "total_degree": [d for _, d in Graph.degree()],
                "weighted_in_degree": [d for _, d in Graph.in_degree(weight="weight")],
                "weighted_out_degree": [
                    d for _, d in Graph.out_degree(weight="weight")
                ],
            }
        )
        degree_data.to_csv(os.path.join(output_folder, "node_degrees.csv"), index=False)

        # 2. Edge Data
        edge_data = pd.DataFrame(
            [
                {
                    "source": u,
                    "target": v,
                    "weight": d["weight"],
                    "toxicity": d.get("avg_toxicity", 0),
                }
                for u, v, d in Graph.edges(data=True)
            ]
        )
        edge_data.to_csv(os.path.join(output_folder, "edge_data.csv"), index=False)

        # 3. Clustering Data
        clustering_data = pd.DataFrame(
            {
                "node": list(Graph.nodes()),
                "clustering_coefficient": list(nx.clustering(Graph).values()),
            }
        )
        clustering_data.to_csv(
            os.path.join(output_folder, "clustering_coefficients.csv"), index=False
        )

        # 4. Component Data
        components = list(nx.weakly_connected_components(Graph))
        component_data = pd.DataFrame(
            {
                "component_id": range(len(components)),
                "size": [len(c) for c in components],
                "density": [nx.density(Graph.subgraph(c)) for c in components],
            }
        )
        component_data.to_csv(
            os.path.join(output_folder, "component_data.csv"), index=False
        )

        # 5. Community Data (using Louvain)
        try:
            partition = community_louvain.best_partition(Graph.to_undirected())
            community_data = pd.DataFrame(
                {
                    "node": list(partition.keys()),
                    "community_id": list(partition.values()),
                }
            )
            community_data.to_csv(
                os.path.join(output_folder, "community_data.csv"), index=False
            )

            # Community summary
            community_summary = pd.DataFrame(
                [
                    {
                        "community_id": comm_id,
                        "size": len([n for n, c in partition.items() if c == comm_id]),
                        "internal_edges": len(
                            [
                                (u, v)
                                for u, v in Graph.edges()
                                if partition[u] == comm_id and partition[v] == comm_id
                            ]
                        ),
                        "external_edges": len(
                            [
                                (u, v)
                                for u, v in Graph.edges()
                                if partition[u] == comm_id and partition[v] != comm_id
                            ]
                        ),
                    }
                    for comm_id in set(partition.values())
                ]
            )
            community_summary.to_csv(
                os.path.join(output_folder, "community_summary.csv"), index=False
            )
        except Exception as e:
            logger.warning(f"Could not calculate community data: {str(e)}")

        # 6. Path Length Data (for largest component)
        try:
            largest_component = max(nx.weakly_connected_components(Graph), key=len)
            largest_subgraph = Graph.subgraph(largest_component)

            # Sample nodes for efficiency
            sample_size = min(1000, max(50, int(len(largest_subgraph) * 0.02)))
            sampled_nodes = np.random.choice(
                list(largest_subgraph.nodes()), size=sample_size, replace=False
            )

            path_data = []
            for source in sampled_nodes:
                lengths = nx.single_source_shortest_path_length(
                    largest_subgraph, source
                )
                for target, length in lengths.items():
                    path_data.append(
                        {
                            "source": source,
                            "target": target,
                            "shortest_path_length": length,
                        }
                    )

            path_df = pd.DataFrame(path_data)
            path_df.to_csv(os.path.join(output_folder, "path_lengths.csv"), index=False)
        except Exception as e:
            logger.warning(f"Could not calculate path length data: {str(e)}")

        # 7. Centrality Metrics
        try:
            k = min(1000, max(50, int(len(Graph) * 0.02)))
            centrality_data = pd.DataFrame(
                {
                    "node": list(Graph.nodes()),
                    "pagerank": list(nx.pagerank(Graph).values()),
                    "betweenness": list(nx.betweenness_centrality(Graph, k=k).values()),
                }
            )
            centrality_data.to_csv(
                os.path.join(output_folder, "centrality_metrics.csv"), index=False
            )
        except Exception as e:
            logger.warning(f"Could not calculate centrality metrics: {str(e)}")

        # 8. Time-based Summary and Advanced Metrics
        base_metrics = {
            "month_start": month_start.strftime("%Y-%m-%d"),
            "nodes": len(Graph),
            "edges": Graph.number_of_edges(),
            "density": nx.density(Graph),
            "avg_clustering": nx.average_clustering(Graph),
            "num_components": len(components),
            "largest_component_size": len(largest_component),
            "avg_degree": np.mean([d for _, d in Graph.degree()]),
            "max_degree": max([d for _, d in Graph.degree()]),
        }

        # Calculate and save advanced metrics
        advanced_metrics = save_advanced_metrics(Graph, output_folder)

        # Combine base and advanced metrics for time summary
        if advanced_metrics:
            base_metrics.update(advanced_metrics)

        time_summary = pd.DataFrame([base_metrics])
        time_summary.to_csv(
            os.path.join(output_folder, "time_summary.csv"), index=False
        )

    except Exception as e:
        logger.error(f"Error saving data tables: {str(e)}")


def process_network_data():
    """Process network data month by month and save all results."""
    try:
        logger.info("Starting network analysis")
        months = get_all_months()
        total_months = len(months)

        # Create timestamp-based main output directory
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        main_output_dir = os.path.join(OUTPUT_DIR, f"{current_timestamp}_{table_name}")
        os.makedirs(main_output_dir, exist_ok=True)

        # Create all_months_summary for tracking metrics across time
        all_months_summary = []

        for i, month_start in enumerate(months, 1):
            logger.info(f"Processing month {i}/{total_months}: {month_start}")

            # Create month-specific folder
            month_folder = os.path.join(main_output_dir, month_start.strftime("%Y_%m"))
            os.makedirs(month_folder, exist_ok=True)

            # Fetch and process data
            df_month = fetch_monthly_data(month_start)
            logger.info(f"Fetched {len(df_month)} rows for {month_start}")

            # Create graph
            Graph = nx.from_pandas_edgelist(
                df_month,
                "source",
                "target",
                edge_attr=["weight", "sum_toxicity", "avg_toxicity"],
                create_using=nx.DiGraph(),
            )

            # Save all data tables
            save_data_tables(month_start, Graph, month_folder)

            # Calculate and save histogram data
            histogram_data = calculate_histogram_data(Graph, month_start)
            if histogram_data:
                with open(os.path.join(month_folder, "histograms.json"), "w") as f:
                    json.dump(histogram_data, f)

            # Read the time_summary.csv for this month and append to all_months_summary
            month_summary = pd.read_csv(os.path.join(month_folder, "time_summary.csv"))
            all_months_summary.append(month_summary)

            logger.info(f"Completed processing for {month_start}")

        # Combine and save all months summary
        all_months_df = pd.concat(all_months_summary, ignore_index=True)
        all_months_df.to_csv(
            os.path.join(main_output_dir, "all_months_summary.csv"), index=False
        )

        # Save processing metadata
        metadata = {
            "processing_timestamp": current_timestamp,
            "table_name": table_name,
            "total_months_processed": total_months,
            "first_month": months[0].strftime("%Y-%m-%d"),
            "last_month": months[-1].strftime("%Y-%m-%d"),
            "output_directory": main_output_dir,
        }

        with open(os.path.join(main_output_dir, "processing_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"All results saved in: {main_output_dir}")
        return main_output_dir

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    logger.info("Starting network metrics calculation")

    try:
        output_dir = process_network_data()
        logger.info("Processing completed successfully")
        logger.info(f"Results are saved in: {output_dir}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
