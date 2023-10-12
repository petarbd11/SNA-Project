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


# Enhance the graph with node attributes from the nodes CSV
for index, row in df_nodes.iterrows():
    node = row['# index']
    if node in G:
        G.nodes[node][' label'] = row[' label']
        G.nodes[node][' viz'] = eval(row[' viz'])  # Convert string representation to dictionary without ast

# Visualize the graph
plt.figure(figsize=(12, 9))
colors = [f"#{int(G.nodes[node][' viz']['color']['r']):02x}{int(G.nodes[node][' viz']['color']['g']):02x}{int(G.nodes[node][' viz']['color']['b']):02x}" for node in G.nodes()]
sizes = [G.nodes[node][' viz']['size'] * 100 for node in G.nodes()]  # multiplying by 100 for better visibility
positions = {(node): (G.nodes[node][' viz']['position']['x'], G.nodes[node][' viz']['position']['y']) for node in G.nodes()}
nx.draw(G, pos=positions, with_labels=True, node_color=colors, node_size=sizes)

# Compute and display metrics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
avg_degree = sum(dict(G.degree()).values()) / num_nodes
density = nx.density(G)

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Average degree: {avg_degree:.2f}")
print(f"Density: {density:.2f}")

plt.show()

#-----------------------------------------------------------WEEK 2

#QUERY:
#While considering the largest component of your network. Depending on what you prefer/seems more relevant in your graph,  
#a') Compute Average clustering and Transitivity number,
#b') Implement a function computing the transitivity using basic function of networkx

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


#-----------------------------------------------------------WEEK 3
'''
QUERY 3:
Depending on what seems more relevant in your graph of the following local notions
- Betweeness centrality
- Closeness centrality

1) Provide a code computing the given centrality using basic functions of networkx (you are not allowed to use directly nx."what you want").
2) Discuss why you picked this measure and who is the most central in your network based on your choice.
3) Provide the cumulative distribution for this centrality and give a graphical representation of your graph (log-log, log or normal representation as you think it is more relevant).
'''

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
