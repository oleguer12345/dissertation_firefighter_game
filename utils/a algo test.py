import heapq
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import math
#np.random.seed(32) 
import random
from scipy.spatial import KDTree
import copy
from graph_class import Graph




def visualize_graph(graph, protected_vertices):
    # Create a NetworkX graph object
    nx_graph = nx.Graph()

    # Add the vertices and edges to the NetworkX graph
    for vertex in graph.vertices:
        nx_graph.add_node(vertex)

    for vertex in graph.vertices:
        for neighbor in graph.get_neighbors(vertex):
            nx_graph.add_edge(vertex, neighbor)

    # Extract x and y coordinates from vertices
    x_coords = [coord[0] for coord in graph.vertices]
    y_coords = [coord[1] for coord in graph.vertices]

    # Draw the graph using Matplotlib
    plt.figure(figsize=(10, 10))

    # Add edges to the plot
    for edge in nx_graph.edges():
        v1, v2 = edge
        x1, y1 = v1
        x2, y2 = v2
        plt.plot([x1, x2], [y1, y2], color='gray', zorder=1)

    # Create a list of node colors for visualization
    node_colors = ['lightblue' if vertex in protected_vertices else 'lightgray' for vertex in graph.vertices]

    # Get burn times for each node
    burn_times = [round(graph.get_burn_time(vertex),1) for vertex in graph.vertices]
    
    # Get burn times for each node
    #importance = [round(graph.get_flammability(vertex),1) if graph.is_burning(vertex) else 0 for vertex in graph.vertices]

    # Draw the vertices on top of the edges
    plt.scatter(x_coords, y_coords, color=node_colors, s=150)
    
    
    # Add labels with burn times for burning nodes
    for i, vertex in enumerate(graph.vertices):
        #if graph.is_burning(vertex):
        plt.text(x_coords[i], y_coords[i], burn_times[i], color='black', ha='center', va='center')


    # set axes range
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    plt.title('Firefighter Game Visualization')
    plt.show()   

# Create a graph instance
graph = Graph([(0.5, 0.5), (1.0, 2.0), (2.5, 1.0), (3.0, 3.0), (4.0, 0.5), (1.0, 0.5), (1.0, 1.5), (2.0, 0.5), (2.0, 3.5)])

# Add edges between vertices
graph.add_edge((0.5, 0.5), (1.0, 2.0))
graph.add_edge((1.0, 2.0), (2.5, 1.0))
graph.add_edge((2.5, 1.0), (3.0, 3.0))
graph.add_edge((2.5, 1.0), (4.0, 0.5))
graph.add_edge((3.0, 3.0), (4.0, 0.5))
graph.add_edge((0.5, 0.5), (1.0, 0.5))
graph.add_edge((0.5, 0.5), (1.0, 1.5))
graph.add_edge((2.5, 1.0), (1.0, 1.5))
graph.add_edge((2.0, 0.5), (1.0, 0.5))
graph.add_edge((2.0, 0.5), (2.5, 1.0))
#graph.add_edge((1.0, 0.5), (2.5, 1.0))
graph.add_edge((2.0, 0.5), (4.0, 0.5))

# Find the minimum distance between two nodes
start_node = (0.5, 0.5)
target_node = (4.0, 0.5)
path = graph.astar_search(start_node, target_node)

if path is not None:
    print("Minimum distance path:", path)
    print("Minimum distance:", len(path) - 1)  # Subtract 1 to exclude the start node
else:
    print("No path found.")

visualize_graph(graph, path)