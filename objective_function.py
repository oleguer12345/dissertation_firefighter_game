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

            #######PROTECT
            for key, value in policy.items():
                print(key)
                if ((key in graph.get_protected_vertices()) or (graph.get_burn_time(key)>0) or (key in graph.get_defunct_vertices())):
                    pass
                else:
                    print("ts:"+str(time_step))
                    print(value[time_step])
                    print("----------")
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





num_vertices=5
start_period=2
graph=generate_gabriel_graph(num_vertices)
policy=make_dict_for_heur(graph.get_all_vertices(),num_vertices,start_period)
print(policy)
# for key, value in policy.items():
#     print(key)
#     print(value)
print(len(graph.get_all_vertices())-1)
print("###################")
objective_function(graph, policy, start_period, full_viz=True)













