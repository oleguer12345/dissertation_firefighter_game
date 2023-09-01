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
import time

    
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
        #print("---->>"+str(vertex_with_max_value))
        ##PROTECT IT
        #print()
        graph.protect_vertex(vertex_with_max_value)
        return vertex_with_max_value
    
def greedy_l1_protection(graph, current_period, start_period):
    if current_period>=start_period:
        fire_neighbours={}
        l1=graph.get_l1_protection_nodes()
    
        #For all neighbours of the fire
        for vertex in l1:
            #get the one with the highest value (number of connexions)
            fire_neighbours[vertex]=len(graph.get_non_burnt_protected_neighbors(vertex))
        
        try:
            vertex_with_max_value = max(fire_neighbours, key=fire_neighbours.get)
            ##PROTECT IT
            graph.protect_vertex(vertex_with_max_value)
            return vertex_with_max_value
        except:
            pass

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
        return vertex_with_max_value    
                
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
        return vertex_with_max_value
        

        
def construction_heuristic(graph, current_period, start_period):
    if current_period>=start_period:
        #Get vertices to potentially protect
        working_vertices = set()
        
        l0v=set(graph.get_l0_protection_nodes())
        nbv=set(graph.get_non_burnt_vertices())
        dv=set(graph.get_defunct_vertices())
        pv=set(graph.get_protected_vertices())
        working_vertices |= l0v
        working_vertices |= nbv
        working_vertices -= pv
        working_vertices -= dv
        vertex_to_defend=[]
        best_score=0
        
        for vertex in working_vertices:
            temp_graph=copy.deepcopy(graph)
            temp_working_vertices=copy.deepcopy(working_vertices)
            temp_working_vertices.remove(vertex)
            for protected in temp_graph.get_protected_vertices():
                temp_graph.delete_vertex(protected)
            temp_graph.delete_vertex(vertex)
            score=temp_graph.get_shadow_score(temp_working_vertices)

            if (score==0 and current_period>2):
                vertex_to_defend=vertex
                #print("over")
                break
            elif score>best_score:
                best_score=score
                vertex_to_defend=vertex
        
        try:
            graph.protect_vertex(vertex_to_defend)
            return vertex_to_defend
        except:
            pass
            
        #return working_vertices
    
    
def construction_heuristic_l0(graph, current_period, start_period):
    if current_period>=start_period:
        #Get vertices to potentially protect
        working_vertices = set()
        
        l0v=set(graph.get_l0_protection_nodes())
        #nbv=set(graph.get_non_burnt_vertices())
        dv=set(graph.get_defunct_vertices())
        pv=set(graph.get_protected_vertices())
        working_vertices |= l0v
        #working_vertices |= nbv
        working_vertices -= pv
        working_vertices -= dv
        vertex_to_defend=[]
        best_score=0
        
        for vertex in working_vertices:
            temp_graph=copy.deepcopy(graph)
            temp_working_vertices=copy.deepcopy(working_vertices)
            temp_working_vertices.remove(vertex)
            for protected in temp_graph.get_protected_vertices():
                temp_graph.delete_vertex(protected)
            temp_graph.delete_vertex(vertex)
            score=temp_graph.get_shadow_score(temp_working_vertices)

            if (score==0 and current_period>2):
                vertex_to_defend=vertex
                #print("over")
                break
            elif score>best_score:
                best_score=score
                vertex_to_defend=vertex
        
        try:
            graph.protect_vertex(vertex_to_defend)
            return vertex_to_defend
        except:
            pass
            
        #return working_vertices

    
#################################################################
############################ MAIN ###############################
#################################################################

def run_main():
    num_vertices=201
    graph=generate_gabriel_graph(num_vertices)
    start_period=2
    time_step=0
    evolve_fire(graph, graph.get_random_vertex(), time_step)
    
    list_burning_nodes=[0]
    list_protected_nodes=[0]
    list_defunct_nodes=[0]
    list_untouched_nodes=[num_vertices]
    
    ##################################################################
    while graph.is_contaned()==False:
        
        evolve_fire(graph, graph.get_random_vertex(), time_step)                           #<<<-----------------
                                              
        #visualize_graph(graph, graph.get_protected_vertices(), graph.get_burning_vertices(), graph.get_defunct_vertices(), time_step)
        #construction_heuristic(graph, time_step, start_period)
        #greedy_l1_protection(graph, time_step, start_period)
        #greedy_l1_joint_protection(graph, time_step, start_period)  
        #greedy_protection(graph, time_step, start_period)
        construction_heuristic_l0(graph, time_step, start_period)
        # if time_step>=start_period:
        #     visualize_graph_2(graph, graph.get_random_vertex(), construction_heuristic(graph,time_step, start_period))
            
        time_step+=1   
    
        #save values for viz
        n_brn=len(graph.get_burning_vertices())
        n_prot=len(graph.get_protected_vertices())
        n_defct=len(graph.get_defunct_vertices())
        list_burning_nodes.append(n_brn)
        list_protected_nodes.append(n_prot)
        list_defunct_nodes.append(n_defct)
        list_untouched_nodes.append(num_vertices-(n_brn+n_prot+n_defct))

    #nodeeeeee=graph.get_random_vertex()
    #visualize_graph_2(graph, graph.get_random_vertex(),construction_heuristic(graph,time_step, start_period))
    #print(graph.get_distance_to_fire((0.24475897516796485, 0.8774848245340555)))
    
    
    ##################################################################    
    visualize_graph(graph, graph.get_protected_vertices(), graph.get_burning_vertices(), graph.get_defunct_vertices(),"99999")
    visualize_evolution(num_vertices, time_step, list_defunct_nodes, list_burning_nodes, list_protected_nodes, list_untouched_nodes)
    #print(len(graph.get_non_burnt_and_protected_vertices()))
    return len(graph.get_non_burnt_and_protected_vertices())


#run_main()
# results=[]
# for x in range(1):
#     y=run_main()
#     results.append(y)
#     print(x)

# # Calculate statistics
# avg = statistics.mean(results)
# mode = statistics.mode(results)
# maximum = max(results)
# minimum = min(results)

# plt.figure(figsize=(8, 5))


# plt.hist(results, bins=20, edgecolor='white', color='darkturquoise')
# plt.axvline(avg, color='r', linestyle='dashed', linewidth=1.5, label='Average')
# plt.axvline(mode, color='gold', linestyle='dashed', linewidth=1.5, label='Mode')

# plt.xlabel('Nodes saved (%)')
# plt.ylabel('Frequency')
# plt.title('"Shadow" Construction Heuristic average return (over 200 simulations)')
# plt.xlim(0, 100)  # Adjust the limits according to your data

# plt.legend()
# plt.show()



def run_heuristic(heuristic,graph,start_period, max_runtime):
    time_step=0
    policy=[]
    evolve_fire(graph, graph.get_random_vertex(), time_step)
    
    s_time = time.time()
    ##################################################################
    while graph.is_contaned()==False:
        
        curr_time = time.time()
        elapsed_time = curr_time-s_time
        if elapsed_time>max_runtime:
            print(f"Elapsed time: {elapsed_time:.6f} seconds")
            break
        
        evolve_fire(graph, graph.get_random_vertex(), time_step)                   
        
        if heuristic=="gl1":                                  
            node=greedy_l1_protection(graph, time_step, start_period)
        elif heuristic=="gl1j":  
            node=greedy_l1_joint_protection(graph, time_step, start_period)
        elif heuristic=="g":  
            node=greedy_protection(graph, time_step, start_period)
        elif heuristic=="c":
            node=construction_heuristic(graph, time_step, start_period)
        else:
            node=construction_heuristic_l0(graph, time_step, start_period)

        if time_step>=start_period:
            policy.append(node)

        time_step+=1   
    

    policy_set=set(policy)
    all_set=set(graph.get_all_vertices())
    diff=all_set-policy_set
    for d in diff:
        policy.append(d)

    if heuristic=="r":
        policy=graph.get_all_vertices()
    return policy    















