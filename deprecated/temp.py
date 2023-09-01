import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import math
#np.random.seed(32) 
import random
from scipy.spatial import KDTree
from graph_class import Graph

#################################################################
################### FIRE BEHAVIOUR DEF ##########################
#################################################################

# def fire_spread_prob(x, m=3.5, a=1.5):
#     return math.exp(-((x - m) / a) ** 2)


def fire_spread_prob(x, m=3.5, a=2.5, l=1.3):
    return math.exp(-((x - m)**2 / a**2) - l)
                
def evolve_fire(graph, start_node,time_step):
    graph.increment_burn_time()
    # If there are no burnt vertices, start a fire at the given node
    if time_step == 0:
        graph.mark_as_burning(start_node)
    else:
        # SPREAD THE FIRE
        for burning_vertex in graph.get_burning_vertices():
            # Get its neighbors
            neighbors = graph.get_neighbors(burning_vertex)

            # If its neighbors are not already burnt
            for neighbor in neighbors:
                # If the neighbor is not burning and is not protected
                if graph.is_burning(neighbor) is False and graph.is_protected(neighbor) is False and graph.is_defunct(neighbor) is False:
                    # Check if the neighbor should catch fire based on the fire spread probability
                    if random.uniform(0, 1) < fire_spread_prob(graph.get_burn_time(burning_vertex)):
                        #print(fire_spread_prob(graph.get_burn_time(burning_vertex)))
                        graph.mark_as_burning(neighbor)
        # After some time, mark burnt vertices as defunct fire
        for burning_vertex in graph.get_burning_vertices():
            if graph.get_burn_time(burning_vertex)>7:
                graph.mark_as_defunct(burning_vertex)
                
#################################################################
########## GABRIEL GRAPH GENERATING FUNCTIONS ###################
#################################################################

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def is_gabriel(point1, point2, coordinates):
    midpoint = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
    midpoint = tuple(midpoint)
    gabriel_dist =calculate_distance(point1, midpoint)
    is_gabriel=True
    for i in range(len(coordinates)):
        if calculate_distance(coordinates[i], midpoint)<gabriel_dist:
            if coordinates[i]!=(point1 and point2):
                is_gabriel=False
    return is_gabriel



def generate_gabriel_graph(num_points):
    
    start_time = time.time()
    np.random.seed(32323)
    points = np.random.rand(num_points, 2)
    tuples_points = [tuple(lst) for lst in points]
    graph = Graph(tuples_points)
    
    tri = Delaunay(tuples_points)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time X1: {execution_time} seconds")
    # print(tuples_points)
    ################################
    start_time = time.time()
    for simplex in tri.simplices:
        #print("")
        # print(tuples_points)
        # print("-------")
        p1, p2, p3 = map(tuple, points[simplex])
        if is_gabriel(p1, p2, tuples_points)==True:
            graph.add_edge(p1, p2)
        if is_gabriel(p1, p3, tuples_points)==True:
            graph.add_edge(p1, p3)
        if is_gabriel(p2, p3, tuples_points)==True:
            graph.add_edge(p2, p3)
            
            
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time X2: {execution_time} seconds")
    
    return graph

 
#################################################################
################## GRAPH VIZ FUNCTIONS ##########################
#################################################################

def visualize_graph(graph, protected_vertices, burnt_vertices, defunct_vertices):
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
    plt.title('Firefighter Game Visualization')
    plt.show()   
    
    
def visualize_evolution(num_vertices, time_step, list_defunct_nodes, list_burning_nodes, list_protected_nodes, list_untouched_nodes):
    # Define custom colors for the areas
    colors = ['gold', 'crimson', 'cornflowerblue', 'palegreen']
    labels=['Defunct', 'Burning', 'Protected', 'Unburned']

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
    
#################################################################
####################### HUERISTICS DEF ##########################
#################################################################
           
                
def greedy_protection(graph, current_period, start_period):
    if current_period>=start_period:
        fire_neighbours={}
        l0=graph.get_l0_protection_nodes()
    
        #For all neighbours of the fire
        for vertex in l0:
            #get the one with the highest value (number of connexions)
            fire_neighbours[vertex]=len(graph.get_non_burnt_protected_neighbors(vertex))
        
        vertex_with_max_value = max(fire_neighbours, key=fire_neighbours.get)
        print("---->>"+str(vertex_with_max_value))
        ##PROTECT IT
        graph.protect_vertex(vertex_with_max_value)
    
def greedy_l1_protection(graph, current_period, start_period):
    if current_period>=start_period:
        fire_neighbours={}
        l1=graph.get_l1_protection_nodes()
    
        #For all neighbours of the fire
        for vertex in l1:
            #get the one with the highest value (number of connexions)
            fire_neighbours[vertex]=len(graph.get_non_burnt_protected_neighbors(vertex))
        
        vertex_with_max_value = max(fire_neighbours, key=fire_neighbours.get)
        ##PROTECT IT
        graph.protect_vertex(vertex_with_max_value)


def greedy_l2_protection(graph, current_period, start_period):
    if current_period>=start_period:
        fire_neighbours={}
        l2=graph.get_l2_protection_nodes()
    
        #For all neighbours of the fire
        for vertex in l2:
            #get the one with the highest value (number of connexions)
            fire_neighbours[vertex]=len(graph.get_non_burnt_protected_neighbors(vertex))
        
        vertex_with_max_value = max(fire_neighbours, key=fire_neighbours.get)
        ##PROTECT IT
        graph.protect_vertex(vertex_with_max_value)    
        
        
def greedy_l1_joint_protection(graph, current_period, start_period):
    if current_period>=start_period:
        l1=graph.get_l1_protection_nodes()
        
        
        if len(graph.get_protected_vertices())>0:
            try:
                list_1={}
                for vertex in graph.get_protected_vertices():
                    for neighbor in graph.get_non_burnt_protected_neighbors(vertex):
                        if neighbor in graph.get_l1_protection_nodes():
                            list_1[neighbor]=len(graph.get_non_burnt_protected_neighbors(neighbor))
                vertex_with_max_value = max(list_1, key=list_1.get)

            except:
                fire_neighbours={}
                for vertex in l1:
                    fire_neighbours[vertex]=len(graph.get_non_burnt_protected_neighbors(vertex))
                if not fire_neighbours:
                    return None
                vertex_with_max_value = max(fire_neighbours, key=fire_neighbours.get)
                
                
        else:
            fire_neighbours={}
            for vertex in l1:
                fire_neighbours[vertex]=len(graph.get_non_burnt_protected_neighbors(vertex))
            if not fire_neighbours:
                return None    
            vertex_with_max_value = max(fire_neighbours, key=fire_neighbours.get)
            
        graph.protect_vertex(vertex_with_max_value)



    
#################################################################
############################ MAIN ###############################
#################################################################
import time

start_time = time.time()

num_vertices=600
graph=generate_gabriel_graph(num_vertices)
start_period=3
time_step=0
evolve_fire(graph, graph.get_random_vertex(), time_step)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time P1: {execution_time} seconds")

list_burning_nodes=[0]
list_protected_nodes=[0]
list_defunct_nodes=[0]
list_untouched_nodes=[num_vertices]

##################################################################
start_time = time.time()

while graph.is_contaned()==False:
    
    evolve_fire(graph, graph.get_random_vertex(), time_step)                           #<<<-----------------
    greedy_protection(graph, time_step, start_period)                                  #<<<-----------------
    #greedy_l1_joint_protection(graph, time_step, start_period)                        #<<<-----------------
    time_step+=1                                                                       #<<<-----------------
    
    #save values for viz
    n_brn=len(graph.get_burning_vertices())
    n_prot=len(graph.get_protected_vertices())
    n_defct=len(graph.get_defunct_vertices())
    list_burning_nodes.append(n_brn)
    list_protected_nodes.append(n_prot)
    list_defunct_nodes.append(n_defct)
    list_untouched_nodes.append(num_vertices-(n_brn+n_prot+n_defct))

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time P2: {execution_time} seconds")

##################################################################    
start_time = time.time()
    
visualize_graph(graph, graph.get_protected_vertices(), graph.get_burning_vertices(), graph.get_defunct_vertices())
visualize_evolution(num_vertices, time_step, list_defunct_nodes, list_burning_nodes, list_protected_nodes, list_untouched_nodes)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time P3: {execution_time} seconds")







