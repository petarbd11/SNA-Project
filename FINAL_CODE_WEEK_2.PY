#-----------------------------------------------------------WEEK 2
#QUERY:
#While considering the largest component of your network. Depending on what you prefer/seems more relevant in your graph,  
#a') Compute Average clustering and Transitivity number,
#b') Implement a function computing the transitivity using basic function of networkx
#-----------------------------------------------------------


import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load nodes and edges from CSV files
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')
edges_df.columns = edges_df.columns.str.strip()  # Trim column names

# Create the graph
G = nx.from_pandas_edgelist(edges_df, source='# source', target='target')

# Extract the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G_main = G.subgraph(largest_cc)

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
    Temp_sum = sum(clustering(G, i) for i in G.nodes()) #Sam
    return Temp_sum / N

# Custom function to compute transitivity using basic functions of NetworkX
def custom_transitivity(G):
    triangles = sum(nx.triangles(G, node) for node in G.nodes()) / 3
    triplets = sum(d * (d - 1) for n, d in G.degree()) / 2
    return triangles / triplets if triplets != 0 else 0

# Compute average clustering using NetworkX's built-in function
avg_clustering_nx = nx.average_clustering(G_main)

# Compute average clustering using custom function
avg_clustering_custom = average_clustering(G_main)

# Compute transitivity using NetworkX's built-in function
transitivity_nx = nx.transitivity(G_main)

# Compute transitivity using custom function
transitivity_custom = custom_transitivity(G_main)

print(f"Average Clustering (NetworkX): {avg_clustering_nx}")
print(f"Average Clustering (Custom): {avg_clustering_custom}")
print(f"Transitivity (NetworkX): {transitivity_nx}")
print(f"Transitivity (Custom): {transitivity_custom}")

# Visualization (Optional)
nx.draw(G_main, with_labels=True)
plt.show()