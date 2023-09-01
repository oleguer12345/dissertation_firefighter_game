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
import time





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
            time_step+=1
            ########################################################
            
        else:
            evolve_fire(graph, graph.get_random_vertex(), time_step)
            time_step+=1
    
    if full_viz==True:    
        visualize_graph(graph, graph.get_protected_vertices(), graph.get_burning_vertices(), graph.get_defunct_vertices(), time_step)    
            
    saved_vertices=len(graph.get_all_vertices())-len(graph.get_defunct_vertices())-len(graph.get_burning_vertices())

    return saved_vertices



def mutate(parent, max_protected_nodes_per_timestep):
    mutated_parent = parent.copy()  # Create a copy of the parent dictionary
    
    for key, row in mutated_parent.items():
        timestep = 0
        for timestep in range(len(row)):
            num_protected_nodes = sum(row)
            if num_protected_nodes < max_protected_nodes_per_timestep[timestep]:
                if row[timestep] == 0:
                    mutated_parent[key][timestep] = 1
                    break
                
            elif num_protected_nodes > max_protected_nodes_per_timestep[timestep]:
                if row[timestep] == 1:
                    mutated_parent[key][timestep] = 0
                    break
    
    return mutated_parent


def crossover_2point(matrix1, matrix2, protected_nodes_per_timestep):
    if len(matrix1) != len(matrix2):
        raise ValueError("Matrices must have the same number of rows.")

    num_timesteps = len(list(matrix1.values())[0])

    # Choose two distinct crossover points
    crossover_points = sorted(random.sample(range(1, num_timesteps), 2))
    
    crossed_matrix = {}
    for key, row1 in matrix1.items():
        if crossover_points[0] <= key[0] < crossover_points[1]:
            crossed_matrix[key] = row1[:]
        else:
            crossed_matrix[key] = matrix2[key][:]

    # Adjust the number of protected nodes at each timestep to match the protected_nodes_per_timestep list
    for timestep in range(num_timesteps):
        num_protected_nodes = sum(crossed_matrix[key][timestep] for key in crossed_matrix)
        while num_protected_nodes > protected_nodes_per_timestep[timestep]:
            unprotected_nodes = [key for key in crossed_matrix if crossed_matrix[key][timestep] == 1]
            if unprotected_nodes:
                node_to_unprotect = random.choice(unprotected_nodes)
                crossed_matrix[node_to_unprotect][timestep] = 0
                num_protected_nodes -= 1
        while num_protected_nodes < protected_nodes_per_timestep[timestep]:
            protected_nodes = [key for key in crossed_matrix if crossed_matrix[key][timestep] == 0]
            if protected_nodes:
                node_to_protect = random.choice(protected_nodes)
                crossed_matrix[node_to_protect][timestep] = 1
                num_protected_nodes += 1

    return crossed_matrix


def roulette_wheel_selection(population, start_period):
    fitness_scores = [objective_function(graph, population_member, start_period, full_viz=False) for population_member in population]
    total_fitness = sum(fitness_scores)
    
    selected_parents = []
    
    for _ in range(2):  # Select two parents
        pick = random.uniform(0, total_fitness)
        current_sum = 0
        
        for population_member, score in zip(population, fitness_scores):
            current_sum += score
            if current_sum > pick:
                selected_parents.append(population_member)
                break
    
    # Ensure the two selected parents are not equal
    while selected_parents[0] == selected_parents[1]:
        pick = random.randint(0, len(population) - 1)
        selected_parents[1] = population[pick]
    
    return selected_parents, fitness_scores


def get_best_policy(policies, scores):
    max_score_index = scores.index(max(scores))
    best_policy = policies[max_score_index]
    return best_policy



def generate_pnpts_list(n):
    return list(range(n))











####################################
##TEST##############################
####################################


###### SET UP ENVIROMENT
num_vertices=250
population_size=25
num_evolutions=200
starting_period=2
protected_nodes_per_timestep=generate_pnpts_list(num_vertices)
graph=generate_gabriel_graph(num_vertices)
policies=[]
max_runtime=5*60
s_time = time.time()
plus_heur=False



if plus_heur==False:
    #create initial solution
    policy=make_dict_for_heur(graph.get_all_vertices(),num_vertices,starting_period)
    ####
    tttttt=0
else:
    #or give it from cons
    warm_start=run_heuristic("other",graph,starting_period,30)
    policy=make_dict_for_heur(warm_start,num_vertices,starting_period)
    policies.append(policy)
    tttttt=1


####### GENERATE A POPULATION
for _ in range(population_size-tttttt):
    random777709890890 = random.Random()
    n=graph.get_all_vertices()
    n2=[]
    n2=copy.deepcopy(n)
    random777709890890.shuffle(n2)
    p=make_dict_for_heur(n2,num_vertices,starting_period)
    policies.append(p)

                    # for pol in policies:
                    #     for key, value in pol.items():
                    #         print(f"{key} : {value}")
                    #     print("----------")
    #print(policies)

###### GENERATE EVOLUTIION
for _ in range(num_evolutions):
    curr_time = time.time()
    elapsed_time = curr_time-s_time
    if elapsed_time>max_runtime:
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        break

    ###Keep best individual
    new_policies=[]

    scores=roulette_wheel_selection(policies, starting_period)
    top_g=get_best_policy(policies, scores[1])
    
    new_policies.append(top_g)

    
    
    
    ###Create the rest
    for _ in range(population_size-2):

        parents=roulette_wheel_selection(policies, starting_period)

        intermingled = crossover_2point(parents[0][0], parents[0][1], protected_nodes_per_timestep)
        offspring=mutate(intermingled, protected_nodes_per_timestep)
        new_policies.append(offspring)

        
        
        
    ### Add one totally random
    hh=graph.get_all_vertices()
    hh2=copy.deepcopy(hh)
    random777709890890.shuffle(hh2)
    new_policies.append(make_dict_for_heur(hh2,num_vertices,starting_period))
    
    print(sorted(scores[1]))
    # print(new_policies)
    policies=new_policies




