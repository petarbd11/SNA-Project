import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load nodes and edges from CSV files
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')
edges_df.columns = edges_df.columns.str.strip()  # Trim column names

# Create the graph
G = nx.from_pandas_edgelist(edges_df, source='# source', target='target')

# Extract the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G_main = G.subgraph(largest_cc)

# Compute metrics for the largest connected component

# Diameter and Average Shortest Path Length
diameter = nx.diameter(G_main)
avg_shortest_path_length = nx.average_shortest_path_length(G_main)

#APPROACH 2

# Custom function to compute clustering of a node
def clustering(G, node):
    k = G.degree[node]
    if k == 0 or k == 1:
        return 0
    else:
        List_nodes = [s for s in G.nodes()]
        i = List_nodes.index(node)
        A = nx.adjacency_matrix(G)
        A3 = A**3
        triangle = A3[i, i] / 2
        den = k * (k - 1) / 2
        return triangle / den

# Custom function to compute average clustering of the graph
def average_clustering(G):
    N = G.number_of_nodes()
    Temp_sum = sum(clustering(G, i) for i in G.nodes())
    return Temp_sum / N

# Compute average clustering using NetworkX's built-in function
avg_clustering_nx = nx.average_clustering(G_main)

# Compute average clustering using custom function
avg_clustering_custom = average_clustering(G_main)

print(f"Diameter: {diameter}")
print(f"Average Shortest Path Length: {avg_shortest_path_length}")
print(f"Average Clustering (NetworkX): {avg_clustering_nx}")
print(f"Average Clustering (Custom): {avg_clustering_custom}")

# Visualization (Optional)
nx.draw(G_main, with_labels=True)
plt.show()
