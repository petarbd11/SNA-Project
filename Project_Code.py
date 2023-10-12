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
#a) Compute Average distance and Diameter,
#b) Improve the function computing the inverse distance function.
#a') Compute Average clustering and Transitivity number,
#b') Implement a function computing the transitivity using basic function of networkx

#START CODE HERE:

# Extract the largest component of the graph 'G'
largest_component = max(nx.connected_components(G), key=len)
G_largest = G.subgraph(largest_component)

# OPTION 1: Average distance and Diameter calculations

# Compute the Average distance of the largest component
avg_distance = nx.average_shortest_path_length(G_largest)

# Compute the Diameter of the largest component
diameter = nx.diameter(G_largest)

# Function to compute the inverse distance between two nodes
def inverse_distance(graph, node1, node2):
    
    #Compute the inverse distance between two nodes.
    #If nodes are disconnected, return 0.
    
    try:
        distance = nx.shortest_path_length(graph, node1, node2)
        return 1 / distance if distance != 0 else 0
    except nx.NetworkXNoPath:
        return 0

# OPTION 2: Average clustering coefficient and Transitivity calculations

# Compute the Average clustering coefficient of the graph
avg_clustering = nx.average_clustering(G)

# Compute the Transitivity number of the graph
transitivity = nx.transitivity(G)

# Function to compute transitivity using basic functions of NetworkX
def compute_transitivity(graph):
    
    #Compute the transitivity of a graph using basic functions of NetworkX.
    
    triangles = sum(nx.triangles(graph).values()) / 3  # Each triangle counted thrice
    triples = sum(len(list(nx.all_neighbors(graph, node))) * (len(list(nx.all_neighbors(graph, node))) - 1) for node in graph) / 2
    
    return 3 * triangles / triples if triples != 0 else 0

# Compute transitivity using the implemented function
computed_transitivity = compute_transitivity(G)

print("OPTION A:")
print("   Diameter: ", diameter)
print("   Average Distance: ",avg_distance)
print("OPTION B:")
print("   Average Clustering: ",avg_clustering)
print("   Transitivity: ",transitivity)
print("   Computed ransativity: ",computed_transitivity)