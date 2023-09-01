import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import copy
import statistics
import time
##############USER GENERATED#############
from utils.graph_class import Graph
from utils.fire import *
from utils.gabriel_graph import *
from utils.viz import *
from construction_heuristic import *

    
    
def generate_pnpts_list(n):
    return list(range(n))

def print_dens(data):

    data_np = np.array(data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a heatmap using imshow
    heatmap = ax.imshow(data_np, cmap='viridis')

    # Add a colorbar to show the values-to-color mapping
    cbar = plt.colorbar(heatmap)
    
    # Add x and y labels
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Node to Protect')


    # Show the plot
    plt.show()

def objective_function(graph, policy, start_period, full_viz=False):
    graph.reset_graph()
    time_step=0
    evolve_fire(graph, graph.get_random_vertex(), time_step)

    while graph.is_contaned()==False:
        if time_step>=start_period-1 and time_step<=(len(graph.get_all_vertices())-1):
            
            ########################################################
            #######EVOLVE
            evolve_fire(graph, graph.get_random_vertex(), time_step)

            #print(policy)
            #######PROTECT
            for key, value in policy.items():
                #print(time_step)
                if ((key in graph.get_protected_vertices()) or (graph.get_burn_time(key)>0) or (key in graph.get_defunct_vertices())):
                    pass
                else:
                    if value[time_step]==1:
                        graph.protect_vertex(key)
                        break
            time_step+=1
            ########################################################
            
        else:
            evolve_fire(graph, graph.get_random_vertex(), time_step)
            time_step+=1
    
        if full_viz==True:
            visualize_graph(graph, graph.get_protected_vertices(), graph.get_burning_vertices(), graph.get_defunct_vertices(), time_step)    
                
    saved_vertices=len(graph.get_all_vertices())-len(graph.get_defunct_vertices())-len(graph.get_burning_vertices())

    return saved_vertices


def add_ones_with_probabilities(matrix, num1s, probabilities_matrix):
    if len(num1s) != len(matrix[0]) or len(probabilities_matrix) != len(matrix):
        raise ValueError("Input dimensions should match the matrix size")
        

    for col_idx, num_ones in enumerate(num1s):
        if num_ones > len(matrix):
            raise ValueError(f"Number of ones in column {col_idx} exceeds the number of rows")

        # Count existing ones in the column
        current_ones = sum(matrix[row_idx][col_idx] for row_idx in range(len(matrix)))

        #Add ones until the desired number is reached
        while current_ones < num_ones:
            # Find the first row with a zero and a favorable probability
            for row_idx in range(len(matrix)):
                if matrix[row_idx][col_idx] == 0 and random.random() < probabilities_matrix[row_idx][col_idx]:
                    matrix[row_idx][col_idx] = 1
                    current_ones += 1
                    


def policies_to_list(policies):

    new_policies = []
    
    for policy in policies:
        new_policy = []
        for key, value in policy.items():
            new_policy.append(value)
        new_policies.append(new_policy)
    
    
    return new_policies

###########MAIN##################################
# SETTINGS
num_vertices = 100  # Number of rows
protected_nodes_per_timestep = generate_pnpts_list(num_vertices)  # Number of ones for each column
num_ants = 50
start_period=2
num_iterations=200
max_runtime=5*60
s_time = time.time()

graph=generate_gabriel_graph(num_vertices)
prob_matrix = [[0.1] * num_vertices for _ in range(num_vertices)]
evaporation_rate=0.7# 0.5 is not bad

ss=[]
best_pol=[]
best_score=[]


plus_heur=False



if plus_heur==True:
    #or give it from cons
    warm_start=run_heuristic("other",graph,starting_period,30)
    policy=make_dict_for_heur(warm_start,num_vertices,starting_period)
    policies.append(policy)
    tttttt=1


for t in range(num_iterations):
    curr_time = time.time()
    elapsed_time = curr_time-s_time
    if elapsed_time>max_runtime:
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        break

    policies=[]
    scores=[]
    

    #GENERATE ANTS POLICIES
    for ant in range(num_ants-1):
        
        # Generate the initial matrix and probabilities matrix
        ant_policy = [[0] * num_vertices for _ in range(num_vertices)]
        # Add random ones to the matrix based on probabilities
        add_ones_with_probabilities(ant_policy, protected_nodes_per_timestep, prob_matrix)
        
        #Created dic
        result_dict = {}
        
        for coord, row in zip(graph.get_all_vertices(), ant_policy):
            result_dict[coord] = row
        
        policies.append(result_dict)
        scores.append(objective_function(graph, result_dict, start_period, full_viz=False))
    
    if t==0:
        # Generate the initial matrix and probabilities matrix
        ant_policy = [[0] * num_vertices for _ in range(num_vertices)]
        # Add random ones to the matrix based on probabilities
        add_ones_with_probabilities(ant_policy, protected_nodes_per_timestep, prob_matrix)
        
        #Created dic
        result_dict = {}
        
        for coord, row in zip(graph.get_all_vertices(), ant_policy):
            result_dict[coord] = row
        
        policies.append(result_dict)
        scores.append(objective_function(graph, result_dict, start_period, full_viz=False))
    else:
        policies.append(best_pol)
        scores.append(best_score)
        
    
    print(max(scores))
    
    #if max(scores)>max(ss):
    best_pol=policies[scores.index(max(scores))]
    best_score=max(scores)
    
    ss.append(max(scores))

    #UPDATE PHEROMONE
    policies = policies_to_list(policies)
    
    #bjective_function(graph, policies[scores.index(max(scores))], start_period, full_viz=False)

    # Update prob_matrix based on scores
    for i, policy in enumerate(policies):
        for row in range(len(policy)):
            for col in range(len(policy[row])):
                prob_matrix[row][col] += (scores[i]-(min(scores)-1)) * policy[row][col] *0.001
    
    # Evaporatee
    for i in range(len(prob_matrix)):
        for j in range(len(prob_matrix[i])):
            prob_matrix[i][j] /= 1+evaporation_rate
            prob_matrix[i][j] += 0.001
    
    #print(prob_matrix)
    #print_dens(prob_matrix)
    
    
visualize_graph(graph, graph.get_protected_vertices(), graph.get_burning_vertices(), graph.get_defunct_vertices(), "__")    
print(ss)

# #%%

# ###########MAIN##################################
# # SETTINGS
# num_vertices = 100  # Number of rows
# protected_nodes_per_timestep = generate_pnpts_list(num_vertices)  # Number of ones for each column
# num_ants = 30
# start_period=2
# num_iterations=1000000
# max_runtime=10*60
# s_time = time.time()

# graph=generate_gabriel_graph(num_vertices)
# prob_matrix = [[0.1] * num_vertices for _ in range(num_vertices)]
# evaporation_rate=1.2# 0.5 is not bad

# ss=[]
# best_pol=[]
# best_score=[]

# for t in range(num_iterations):
#     curr_time = time.time()
#     elapsed_time = curr_time-s_time
#     if elapsed_time>max_runtime:
#         print(f"Elapsed time: {elapsed_time:.6f} seconds")
#         break

#     policies=[]
#     scores=[]
    

#     #GENERATE ANTS POLICIES
#     for ant in range(num_ants):
        
#         # Generate the initial matrix and probabilities matrix
#         ant_policy = [[0] * num_vertices for _ in range(num_vertices)]
#         # Add random ones to the matrix based on probabilities
#         add_ones_with_probabilities(ant_policy, protected_nodes_per_timestep, prob_matrix)
        
#         #Created dic
#         result_dict = {}
        
#         for coord, row in zip(graph.get_all_vertices(), ant_policy):
#             result_dict[coord] = row
        
#         policies.append(result_dict)
#         scores.append(objective_function(graph, result_dict, start_period, full_viz=False))
    
#     # if t==0:
#     #     # Generate the initial matrix and probabilities matrix
#     #     ant_policy = [[0] * num_vertices for _ in range(num_vertices)]
#     #     # Add random ones to the matrix based on probabilities
#     #     add_ones_with_probabilities(ant_policy, protected_nodes_per_timestep, prob_matrix)
        
#     #     #Created dic
#     #     result_dict = {}
        
#     #     for coord, row in zip(graph.get_all_vertices(), ant_policy):
#     #         result_dict[coord] = row
        
#     #     policies.append(result_dict)
#     #     scores.append(objective_function(graph, result_dict, start_period, full_viz=False))
#     # else:
#     # policies.append(best_pol)
#     # scores.append(best_score)
    
    
#     print(max(scores))
    
#     #if max(scores)>max(ss):
#     # best_pol=policies[scores.index(max(scores))]
#     # best_score=max(scores)
    
#     ss.append(max(scores))
#     #print(policies)
#     #UPDATE PHEROMONE
#     policies = policies_to_list(policies)
#     #print(policies)
#     #bjective_function(graph, policies[scores.index(max(scores))], start_period, full_viz=False)

#     # Update prob_matrix based on scores
#     for i, policy in enumerate(policies):
#         for row in range(len(policy)):
#             for col in range(len(policy[row])):
#                 prob_matrix[row][col] += (scores[i]-(min(scores)-1)) * policy[row][col] *0.001
    
#     # Evaporatee
#     for i in range(len(prob_matrix)):
#         for j in range(len(prob_matrix[i])):
#             prob_matrix[i][j] /= 1+evaporation_rate
#             prob_matrix[i][j] += 0.001
    
#     #print(prob_matrix)
#     #print_dens(prob_matrix)
    
    
# visualize_graph(graph, graph.get_protected_vertices(), graph.get_burning_vertices(), graph.get_defunct_vertices(), "__")    
# print(ss)

#%%

import matplotlib.pyplot as plt

# col1 = [45,44,56,48,42,42,42,40,57,42,47,36,39,41,40,46,41,39,39,45,44,47,54,57,54,42,41,45,40,53,44,42,46,47,42,44,50,65,44,70,59,47,48,46,37,45,63,66,43,43,68,46,44,45,41,73,47,59,40,57,63,44,63,66,58,67,59,47,55,67,60,67,67,53,70,77,47,66,61,70,54,70,58,68,71,70,68,75,72,70,69,75,70,48,73,48,70,69,60,69]
# col2 = [44,34,41,38,37,41,46,45,39,70,50,40,45,48,43,48,45,41,59,40,41,60,41,63,49,42,68,54,44,64,73,62,67,40,46,69,68,50,43,74,50,59,60,68,71,76,75,52,73,71,66,46,63,61,63,75,67,67,67,70,70,59,74,75,69,75,75,66,75,68,74,69,65,77,63,76,76,79,77,76,90,75,73,75,76,92,67,74,76,76,92,91,76,91,66,91,91,91,77,91]
# col3 = [47,43,38,41,40,43,39,47,42,38,46,45,49,51,41,58,52,47,50,44,43,52,40,44,43,41,60,44,47,43,51,45,62,63,43,59,44,43,48,48,45,47,43,48,47,43,46,38,43,58,51,47,45,47,73,49,43,41,43,41,76,52,41,67,53,62,43,61,57,43,58,75,67,70,57,70,63,58,45,44,44,46,61,58,66,42,45,65,44,44,58,47,42,43,46,63,48,44,52,46]

plt.figure(figsize=(15, 5))

def plot_with_trend(data, ax, col):
    x = np.arange(len(data))
    trend = np.polyfit(x, data, 1)
    trend_line = np.polyval(trend, x)
    
    ax.plot(x, data, label='Nodes saved')
    ax.plot(x, trend_line, label=f'Trend', linestyle='--')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(data)-1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Nodes saved')
    ax.legend()

plt.subplot(1, 3, 1)
plot_with_trend(col1, plt.gca(), 1)
plt.title('PER=0.3')

plt.subplot(1, 3, 2)
plot_with_trend(col2, plt.gca(), 2)
plt.title('PER=0.7')

plt.subplot(1, 3, 3)
plot_with_trend(col3, plt.gca(), 3)
plt.title('PER=1.2')

plt.tight_layout()
plt.show()
