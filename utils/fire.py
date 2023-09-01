import math
import random


#################################################################
################### FIRE BEHAVIOUR DEF ##########################
#################################################################

def fire_spread_prob(x, m=3, a=1.735, l=0.1):
    return math.exp(-((x - m)**2 / a**2) - l)
                
def evolve_fire(graph, start_node,time_step):
    
    myRandom = random.Random(20)
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
                    #random.seed(32)
                    if myRandom.uniform(0, 1) < fire_spread_prob(graph.get_burn_time(burning_vertex)):
                        #print(fire_spread_prob(graph.get_burn_time(burning_vertex)))
                        graph.mark_as_burning(neighbor)
        # After some time, mark burnt vertices as defunct fire
        for burning_vertex in graph.get_burning_vertices():
            if graph.get_burn_time(burning_vertex)>7:
                graph.mark_as_defunct(burning_vertex)