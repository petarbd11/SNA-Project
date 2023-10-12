import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#-----------------------------------------------------------WEEK 1
#QUERY 1:
#a) Pick a social network among the one proposed on luiss.learn
#b) Implement it in Python.
#c) Draw the graph 
#d) Compute the number of nodes,edges, average degree and the density. Comment.

#START CODE HERE:

#Be careful if the network that you have picked is directed or not.
#IMPORTING DATABASE
edges_filename = "edges.csv"
nodes_filename = "nodes.csv"

df_edges = pd.read_csv(edges_filename)
df_nodes = pd.read_csv(nodes_filename)


# Create the graph using the edges CSV
G = nx.from_pandas_edgelist(df_edges, '# source', ' target')  # Adjusted column name
# Closeness Centrality
def compute_closeness_centrality(graph):
    closeness_centrality = {}
    for node in graph.nodes():
        total_distance = sum(nx.shortest_path_length(graph, node, target) for target in graph.nodes() if target != node)
        num_nodes = len(graph.nodes()) - 1  # Excluding the node itself
        closeness_centrality[node] = num_nodes / total_distance if total_distance != 0 else 0
    return closeness_centrality

closeness_centrality_values = compute_closeness_centrality(G)
most_central_node_closeness = max(closeness_centrality_values, key=closeness_centrality_values.get)

# Betweenness Centrality
def compute_betweenness_centrality(graph):
    return nx.betweenness_centrality(graph)

betweenness_centrality_values = compute_betweenness_centrality(G)
most_central_node_betweenness = max(betweenness_centrality_values, key=betweenness_centrality_values.get)

# Cumulative Distribution Visualization
import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_distribution(values, title):
    sorted_values = np.sort(list(values))
    yvals = np.arange(1, len(sorted_values) + 1) / float(len(sorted_values))
    
    plt.figure(figsize=(10, 7))
    plt.plot(sorted_values, 1 - yvals, marker='o', linestyle='-', markersize=4)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title)
    plt.xlabel('Centrality Value')
    plt.ylabel('Fraction of Nodes with Higher Centrality')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()

plot_cumulative_distribution(closeness_centrality_values.values(), 'Cumulative Distribution of Closeness Centrality')
plot_cumulative_distribution(betweenness_centrality_values.values(), 'Cumulative Distribution of Betweenness Centrality')

