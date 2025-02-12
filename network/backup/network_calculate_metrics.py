import pandas as pd
import networkx as nx
import numpy as np
import community as community_louvain  # Import community detection library


def calculate_network_metrics(df):
    # Dictionary to hold weekly metrics
    weekly_metrics = {}

    # Process each week
    for week in df["Week"].unique():
        # Filter data for the current week
        weekly_data = df[df["Week"] == week]

        # Create a directed graph from the weekly data
        G = nx.from_pandas_edgelist(
            weekly_data,
            "Source",
            "Target",
            edge_attr="Weight",
            create_using=nx.DiGraph(),
        )

        # Calculate metrics
        metrics = {
            "average_path_length": np.nan,
            "diameter": np.nan,
            "density": nx.density(G),
            "connected_components": nx.number_strongly_connected_components(G),
            "modularity": np.nan,  # To be calculated below
            "modularity_classes": 0,  # To count the number of unique modularity classes
            "transitivity": nx.transitivity(G),
            "assortativity": nx.degree_assortativity_coefficient(G),
        }

        # Calculating average path length and diameter only if the graph is connected
        if nx.is_weakly_connected(G):
            lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="Weight"))
            all_lengths = [
                length
                for target_lengths in lengths.values()
                for length in target_lengths.values()
            ]
            if all_lengths:
                metrics["average_path_length"] = sum(all_lengths) / len(all_lengths)
                metrics["diameter"] = max(all_lengths)

        # Community detection and modularity calculation
        if len(G) > 0:  # Ensure the graph is not empty
            partition = community_louvain.best_partition(
                G.to_undirected(), weight="Weight"
            )
            modularity = community_louvain.modularity(partition, G.to_undirected())
            metrics["modularity"] = modularity
            # Calculate the number of unique modularity classes
            metrics["modularity_classes"] = len(set(partition.values()))

        # Assigning weekly metrics
        weekly_metrics[week] = metrics

    return weekly_metrics


# Example usage
# Load your data
csv_file_path = "path_to_your_csv_file.csv"  # Specify the path to your CSV file
df = pd.read_csv(csv_file_path)

# Calculate metrics
metrics_by_week = calculate_network_metrics(df)
for week, metrics in metrics_by_week.items():
    print(f"Week {week}: {metrics}")
