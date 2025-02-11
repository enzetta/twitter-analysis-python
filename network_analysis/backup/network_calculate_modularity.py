import pandas as pd
import networkx as nx
import community as community_louvain


def calculate_modularity(csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Create a directed graph from the dataframe
    G = nx.from_pandas_edgelist(
        df, "Source", "Target", edge_attr="Weight", create_using=nx.DiGraph()
    )

    # Compute the best partition (community detection)
    partition = community_louvain.best_partition(G.to_undirected(), weight="Weight")

    # Calculate the modularity
    modularity = community_louvain.modularity(
        partition, G.to_undirected(), weight="Weight"
    )
    return modularity


# Example usage
csv_file_path = "path_to_your_csv_file.csv"  # Specify the path to your CSV file
modularity = calculate_modularity(csv_file_path)
print("Modularity of the network:", modularity)
