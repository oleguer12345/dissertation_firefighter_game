from scipy.spatial import Delaunay
import random
import numpy as np
from utils.graph_class import Graph
import math

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
    
    #32324 oringinal seed#
    #new 982
    np.random.seed()
    points = np.random.rand(num_points, 2)
    tuples_points = [tuple(lst) for lst in points]
    graph = Graph(tuples_points)
    
    tri = Delaunay(tuples_points)
    ################################
    for simplex in tri.simplices:
        p1, p2, p3 = map(tuple, points[simplex])
        if is_gabriel(p1, p2, tuples_points)==True:
            graph.add_edge(p1, p2)
        if is_gabriel(p1, p3, tuples_points)==True:
            graph.add_edge(p1, p3)
        if is_gabriel(p2, p3, tuples_points)==True:
            graph.add_edge(p2, p3)
            
            
    
    return graph
