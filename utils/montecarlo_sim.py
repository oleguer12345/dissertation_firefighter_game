# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:14:56 2023

@author: S2461914
"""
import random
import math
import statistics

def fire_spread_prob(x, m=4, a=2.23, l=0.1):
    return math.exp(-((x - m)**2 / a**2) - l)


# def fire_spread_prob(x, m=3, a=3, l=0.1):
#     return math.exp(-((x - m)**2 / a**2) - l)

times=[]


for _ in range(1000000):
    
    x=1
    keep=True
    while keep is True:
        random_number = random.random()
        #print(fire_spread_prob(x))        
        #print(random_number)
        #print("--------")
        if fire_spread_prob(x)>random_number or x>8:
            keep=False
        else:
            x+=1
    if x<=8:
        times.append(x)
    
    
print(statistics.mean(times))




# #################################

# # Given values
# n = 500  # Number of nodes
# p = 0.025  # Edge probability

# average_neighbors = (n - 1) * p
# print(f"The average number of neighbors for any node: {average_neighbors:.2f}")

