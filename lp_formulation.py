import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#np.random.seed(32) 
import random
#from scipy.spatial import KDTree
import copy
import statistics
##############USER GENERATED#############
from utils.graph_class import Graph
from utils.fire import *
from utils.gabriel_graph import *
from utils.viz import *
from construction_heuristic import *

import xpress as xp
import numpy as np



def transofrm_dict_int(original_dict,node_to_assign_1):
    # Create a mapping between coordinates and unique integers
    coord_to_int = {}
    next_int = 1
    
    new_dict = {}
    
    for key, value in original_dict.items():
        # Process key
        if key not in coord_to_int:
            coord_to_int[key] = next_int
            next_int += 1
        new_key = coord_to_int[key]
        
        # Process values
        new_values = []
        for coord_pair in value:
            if coord_pair not in coord_to_int:
                coord_to_int[coord_pair] = next_int
                next_int += 1
            new_values.append(coord_to_int[coord_pair])
        
        new_dict[new_key] = new_values
    
    # Assign value 1 to the specified node
    if node_to_assign_1 in coord_to_int:
        node_value_1 = coord_to_int[node_to_assign_1]
        new_dict[node_value_1] = [1]
    
    return new_dict



def optimize_fire_spread(G, T, N, spread_buffer):
    # Create a new MILP (Mixed-Integer Linear Programming) problem
    prob = xp.problem()
    #prob.interrupt(xp.stop_timelimit, 10)
    prob.controls.maxtime = -220
    
    #################################################################
    # Sets
    x = range(1, G + 1)
    t = range(T + 1)
    N_dict = N
    
    # Variables
    b = {}
    d = {}
    f = {}
    z = {}
    sb=spread_buffer
    dt=8
    
    
    
    for x_i in x:
        for t_i in t:
            b[x_i, t_i] = xp.var(name=f'b_{x_i}_{t_i}',vartype=xp.binary)
            prob.addVariable(b[x_i, t_i])
            d[x_i, t_i] = xp.var(name=f'd_{x_i}_{t_i}',vartype=xp.binary)
            prob.addVariable(d[x_i, t_i])
            f[x_i, t_i] = xp.var(name=f'f_{x_i}_{t_i}',vartype=xp.binary)
            prob.addVariable(f[x_i, t_i])
            z[x_i, t_i] = xp.var(name=f'z_{x_i}_{t_i}',vartype=xp.binary)
            prob.addVariable(z[x_i, t_i])

    #################################################################
    # Objective function
    prob.setObjective(sum(b[x_i, T] for x_i in x), sense = xp.minimize)
    
    #################################################################
    # Constraints
    for x_i in x:
        prob.addConstraint(b[x_i, 0] == (1 if x_i == 1 else 0))
        prob.addConstraint(d[x_i, 0] == 0)
        prob.addConstraint(f[x_i, 0] == 0)
        prob.addConstraint(z[x_i, 0] == 0)
        for t_i in t[1:]:
            if t_i < sb:
                prob.addConstraint(f[x_i, t_i]==0)
            if t_i < sb+dt:
                prob.addConstraint(z[x_i, t_i]==0)

    ##BASIC FIRE
    for t_i in t[1:]:
        for x_i in x:
            neighbors = N_dict[x_i]
            
            for y in neighbors:
                prob.addConstraint(b[x_i, t_i] + d[x_i, t_i] - f[y, t_i] + z[y, t_i] >= 0)

            prob.addConstraint(b[x_i, t_i] + d[x_i, t_i] <= 1)

            prob.addConstraint(b[x_i, t_i] - b[x_i, t_i-1] >= 0)

            #prob.addConstraint(d[x_i, t_i] - d[x_i, t_i-1] >= 0)

    for t_i in t[1:]:
        prob.addConstraint(sum(d[x_i, t_i] - d[x_i, t_i-1] for x_i in x) <= 1)
    
    for x_i in x:
        for t_i in range(sb, T + 1):
            prob.addConstraint(f[x_i, t_i] == b[x_i, t_i - sb])
            prob.addConstraint(f[x_i, t_i] <= b[x_i, t_i])
            
            
    for x_i in x:
        for t_i in range(dt, T + 1):
            prob.addConstraint(z[x_i, t_i] == f[x_i, t_i - dt])
            prob.addConstraint(z[x_i, t_i] <= f[x_i, t_i])
    
    
    #################################################################
    # Add initial solution coming form construction
    # Set initial solutions using p.addmipsol([])
    # initial_solution = []
    
    # for x_i in x:
    #     for t_i in t:
    #         # Initialize b, d, f, and z based on the provided initial_solution_matrix
    #         initial_b = initial_solution_matrix['b'][x_i - 1][t_i - 1]
    #         initial_d = initial_solution_matrix['d'][x_i - 1][t_i - 1]
    #         initial_f = initial_solution_matrix['f'][x_i - 1][t_i - 1]
    #         initial_z = initial_solution_matrix['z'][x_i - 1][t_i - 1]
    
    #         initial_solution.append(b[x_i, t_i] == initial_b)
    #         initial_solution.append(d[x_i, t_i] == initial_d)
    #         initial_solution.append(f[x_i, t_i] == initial_f)
    #         initial_solution.append(z[x_i, t_i] == initial_z)
    
    # prob.addmipsol(initial_solution)
    
    
    
    #################################################################
    # Solve the problem
    prob.solve()
    
    #results = {x_i: {'b': [], 'd': [], 'f': [], 'z': []} for x_i in x}
    results = {x_i: {'d': []} for x_i in x}
    
    print("#############################")
    for x_i in x:
        for t_i in t:
            # results[x_i]['b'].append(prob.getSolution(b[x_i, t_i]))
            results[x_i]['d'].append(prob.getSolution(d[x_i, t_i]))
            # results[x_i]['f'].append(prob.getSolution(f[x_i, t_i]))
            # results[x_i]['z'].append(prob.getSolution(z[x_i, t_i]))
            
    # # Print the results
    # if prob.getProbStatus() == 1:
    #     print("Worked.")
    # else:
    #     print("Optimization failed.")

    #return results

    #################################################################
###Generate graph
num_vertices=250
graph=generate_gabriel_graph(num_vertices)
result_dict = {}
for x in graph.get_all_vertices():
    neighbors = set(graph.get_neighbors(x))
    result_dict[x] = list(neighbors)

###Create lp
G = num_vertices  # Number of nodes in the graph G
T = num_vertices  # Number of time-steps
node_to_assign_one = graph.get_random_vertex()
N = transofrm_dict_int(result_dict,node_to_assign_one)  # Neighbors of each node in G
spread_buffer = 3

def print_dict(data):
    for key, value in data.items():
        print(f"{key}:")
        for inner_key, inner_value in value.items():
            print(f"  {inner_key}: {np.absolute(inner_value)}")


optimize_fire_spread(G, T, N, spread_buffer)
#print_dict(optimize_fire_spread(G, T, N, spread_buffer))


#visualize_graph(graph, graph.get_protected_vertices(), graph.get_burning_vertices(), graph.get_defunct_vertices(),"99999")
#print(result_dict)
#print(N)
