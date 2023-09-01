import networkx as nx
import matplotlib.pyplot as plt


from utils.graph_class import Graph


#################################################################
################## GRAPH VIZ FUNCTIONS ##########################
#################################################################

def visualize_graph(graph, protected_vertices, burnt_vertices, defunct_vertices, timestep):
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
    node_colors = ['lightblue' if vertex in protected_vertices else 'red' if vertex in burnt_vertices else 'gold' if vertex in defunct_vertices else 'lightgray' for vertex in graph.vertices]

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
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Firefighter Game Visualization'+str(timestep))
    plt.show()   
    
def visualize_graph_2(graph, origin, neighbors):
    # Create a NetworkX graph object
    nx_graph = nx.Graph()

    # Add the vertices and edges to the NetworkX graph
    for vertex in graph.vertices:
        nx_graph.add_node(vertex)
    

    for vertex in graph.vertices:
        #print("-->>"+str(vertex))
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
    node_colors = ['lightblue' if vertex in neighbors else 'gold' for vertex in graph.vertices]
    #node_colors = ['lightblue' if vertex==origin else 'red' if vertex in neighbors else 'gold' for vertex in graph.vertices]

    # Get burn times for each node
    burn_times = [round(graph.get_distance_to_fire(vertex),1) for vertex in graph.vertices]
    
    # Get burn times for each node
    #importance = [round(graph.get_flammability(vertex),1) if graph.is_burning(vertex) else 0 for vertex in graph.vertices]

    # Draw the vertices on top of the edges
    plt.scatter(x_coords, y_coords, color=node_colors, s=150)
    
    
    # Add labels with burn times for burning nodes
    for i, vertex in enumerate(graph.vertices):
        #if graph.is_burning(vertex):
        plt.text(x_coords[i], y_coords[i], burn_times[i], color='black', ha='center', va='center')


    # set axes range
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Firefighter Game Visualization')
    plt.show()   
    
def visualize_evolution(num_vertices, time_step, list_defunct_nodes, list_burning_nodes, list_protected_nodes, list_untouched_nodes):
    # Define custom colors for the areas
    colors = ['gold', 'crimson', 'cornflowerblue', 'palegreen']
    labels=['Defunct', 'Burning', 'Protected', 'Unburned']
    # Draw the graph using Matplotlib
    plt.figure(figsize=(10, 10))
    
    # Plotting the normalized area graph with custom colors
    plt.stackplot(list(range(0, time_step+1)), list_defunct_nodes, list_burning_nodes, list_protected_nodes, list_untouched_nodes, labels=labels, colors=colors)



    # Customizing the graph
    plt.title('Fire evolution')
    plt.xlabel('Time period')
    plt.ylabel('Vertices Burnt (count)')
    plt.legend(loc='upper left')

    plt.xlim(0, time_step)  # Adjust the limits according to your data
    plt.ylim(0, num_vertices)  # Adjust the limits according to your data

    # Displaying the graph
    plt.show()