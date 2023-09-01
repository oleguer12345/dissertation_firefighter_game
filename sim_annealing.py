import math
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



def make_dict_for_heur(data, timesteps, start_period):
    start_period=start_period-1
    matrix = [[0] * timesteps for _ in range(len(data))]
    
    for timestep in range(timesteps):
        for node_index, (node1, node2) in enumerate(data):
            if timestep >= node_index+start_period:
                matrix[node_index][timestep] = 1
    
    result_dict = {}
    for (node1, node2), row in zip(data, matrix):
        result_dict[(node1, node2)] = row
    
    # for key, value in result_dict.items():
    #     print(f"{key} : {value}")
        
    return result_dict


def policies_to_list(policies):

    new_policies = []
    
    for policy in policies:
        new_policy = []
        for key, value in policy.items():
            new_policy.append(value)
        new_policies.append(new_policy)
    
    
    return new_policies

def perturb_matrix(matrix,perc_cols_modif):
    num_rows = len(matrix)
    
    n_cols_mod=perc_cols_modif*num_rows
    
    for x in range(math.ceil(n_cols_mod)):
        row_to_modif = random.randint(0, num_rows-1)
        column_n = [row[row_to_modif] for row in matrix]
        random.shuffle(column_n)
        for i in range(len(matrix)):
            matrix[i][row_to_modif] = column_n[i]

    return matrix


def generate_pnpts_list(n):
    return list(range(n))



def acceptance_probability(current_obj, new_obj, temperature):
    # if new_obj < current_obj:
    #     return 1.0
    return math.exp((new_obj-current_obj) / temperature)

def quadratic_multiplicative_schedule(initial_temperature, iteration, max_iterations, cooling_factor):
    return initial_temperature / (1 + cooling_factor * (iteration / max_iterations) ** 2)



###########MAIN##################################
# SETTINGS
num_vertices = 250  # Number of rows
protected_nodes_per_timestep = generate_pnpts_list(num_vertices)  # Number of ones for each column
start_period=2
max_runtime=50*60
s_time = time.time()
graph=generate_gabriel_graph(num_vertices)
plus_heur=False



if plus_heur==False:
    #create initial solution
    policy=make_dict_for_heur(graph.get_all_vertices(),num_vertices,start_period)
    ####
else:
    #or give it from cons
    warm_start=run_heuristic("other",graph,start_period,30)
    policy=make_dict_for_heur(warm_start,num_vertices,start_period)
saved_ver=objective_function(graph, policy, start_period, full_viz=False)



#config 2
current_solution = policy
current_obj = saved_ver
initial_temp = 5.0
max_iters = 500
cooling_factor = 50  #Adjust this value as needed
temperature = initial_temp

#################################################################################
evolution_sol=[]
evolution_temp = []
iteration=0
#start iterating
while temperature > 0.0000000000001:
    curr_time = time.time()
    elapsed_time = curr_time-s_time
    if elapsed_time>max_runtime:
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        break
    
    print(temperature)
    # print(temperature)
    new_policy=copy.deepcopy(current_solution)
    keys=[]
    for pol in new_policy:
        keys.append(pol)
        
    #perturbate matrix
    list_pol=policies_to_list([new_policy])[0]
    perturb_matrix(list_pol,0.2)
    
    # for p in new_policy.items():
    #     print(p)
    # print("#######")
    
    #back to dict
    result_dict = dict(zip(keys, list_pol))

    #how good it is    
    new_saved_ver=objective_function(graph, new_policy, start_period, full_viz=False)
    
    print(current_obj)
    #print(new_saved_ver)
    # print(acceptance_probability(current_obj, new_saved_ver, temperature))
    if acceptance_probability(current_obj, new_saved_ver, temperature) > random.random():
        current_solution = new_policy
        current_obj = new_saved_ver
    
    # print("#########")
    evolution_sol.append(current_obj)
    evolution_temp.append(temperature)  
    
    temperature = quadratic_multiplicative_schedule(initial_temp, iteration, max_iters, cooling_factor)
    iteration += 1
# Create the first plot for objective function values
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.plot(evolution_sol, color='blue', label='Objective Function')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.legend(loc='upper left')

# Create a twin Axes sharing the xaxis for temperature values
ax2 = plt.gca().twinx()
ax2.plot(evolution_temp, color='red', label='Temperature')
ax2.set_ylabel('Temperature')
ax2.legend(loc='upper right')

plt.show()
    
