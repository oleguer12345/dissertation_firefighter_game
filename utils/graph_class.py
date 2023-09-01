import heapq
#np.random.seed(32) 
import random


class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adjacency_list = {vertex: [] for vertex in vertices}
        self.burnt_vertices = set()  # Set to store the burnt vertices
        self.protected_vertices = set()
        self.defunct_vertices = set()
        self.burn_times = {}  # Dictionary to store burn times
        
    def reset_graph(self):
        self.burnt_vertices.clear() # Set to store the burnt vertices
        self.protected_vertices.clear()
        self.defunct_vertices.clear()
        self.burn_times.clear()  # Dictionary to store burn times
        
    def delete_vertex(self, vertex):
        if vertex in self.adjacency_list:
            # Remove the vertex from adjacency lists of all connected vertices
            for adjacent_vertex in self.adjacency_list[vertex]:
                self.adjacency_list[adjacent_vertex].remove(vertex)
            # Remove the vertex from the graph's adjacency list
            del self.adjacency_list[vertex]
            # Remove the vertex from the self.vertices list
            self.vertices.remove(vertex)
            # Remove the vertex from other sets and dictionaries
            self.burnt_vertices.discard(vertex)
            self.protected_vertices.discard(vertex)
            self.defunct_vertices.discard(vertex)
            self.burn_times.pop(vertex, None)
        else:
            print("The vertex does not exist in the graph.")
    
    def get_random_vertex(self):
        rng_state = random.getstate()
        random.seed(7862)
        #50, 100: 32
        #250: 7862
        choice=random.choice(self.vertices)
        random.setstate(rng_state)

        return choice 
    
    def add_edge(self, u, v):
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)
        
    def remove_edge(self, u, v):
        if u in self.adjacency_list and v in self.adjacency_list:
            self.adjacency_list[u].remove(v)
            self.adjacency_list[v].remove(u)
        else:
            print("One or both of the vertices do not exist in the graph.")
        
    def mark_as_burning(self, vertex):
        self.burnt_vertices.add(vertex)
        self.burn_times[vertex] = 0  # Set the initial burn time to 0 for the newly burnt vertex
        
    def mark_as_defunct(self, vertex):
        self.defunct_vertices.add(vertex)
        self.burnt_vertices.discard(vertex)
        

    def get_burn_time(self, vertex):
        return self.burn_times.get(vertex, 0)  # Return the burn time of the vertex if it exists, else None
        
    def increment_burn_time(self):
        for vertex in self.burnt_vertices:
            self.burn_times[vertex] += 1

    def is_burning(self, vertex):
        return vertex in self.burnt_vertices
    
    def is_defunct(self, vertex):
        return vertex in self.defunct_vertices
    
    def protect_vertex(self, vertex):
        self.protected_vertices.add(vertex)
        self.burnt_vertices.discard(vertex)

    def is_protected(self, vertex):
        return vertex in self.protected_vertices
        
    def get_all_vertices(self):
        return self.vertices

    def get_defunct_vertices(self):
        return list(self.defunct_vertices)

    def get_burning_vertices(self):
        return list(self.burnt_vertices)

    
    def get_non_burnt_and_protected_vertices(self):
        non_burnt_vertices = []
        for vertex in self.vertices:
            if not self.is_burning(vertex) and not self.is_protected(vertex):
                non_burnt_vertices.append(vertex)
        return non_burnt_vertices

    def get_non_burnt_vertices(self):
        return list(set(self.vertices) - self.burnt_vertices)    

    def get_protected_vertices(self):
        return list(self.protected_vertices)
    
    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]
    
    def get_non_burnt_neighbors(self, vertex):
        non_burnt_neighbors = []
        neighbors = self.get_neighbors(vertex)
        for neighbor in neighbors:
            if not self.is_burning(neighbor):
                non_burnt_neighbors.append(neighbor)
        return non_burnt_neighbors
    
    def get_non_burnt_protected_neighbors(self, vertex):
        
        non_burnt_neighbors = []
        neighbors = self.get_neighbors(vertex)
        for neighbor in neighbors:
            if not self.is_burning(neighbor) and not self.is_protected(neighbor):
                non_burnt_neighbors.append(neighbor)
        return non_burnt_neighbors
    
    def get_l0_protection_nodes(self):
        l0_protection_nodes = set()  # Use a set instead of a list
        for burning_vertex in self.get_burning_vertices():
            if self.get_burn_time(burning_vertex) == 0:
                l0_protection_nodes.add(burning_vertex)
            neighbors = self.get_neighbors(burning_vertex)
            for neighbor in neighbors:
                if self.get_burn_time(neighbor) == 0 and not self.is_protected(neighbor):
                    l0_protection_nodes.add(neighbor)
        return list(l0_protection_nodes)
    
    
    def get_l1_protection_nodes(self):
        l1_protection_nodes = set()  # Use a set instead of a list
        for vertex in self.get_l0_protection_nodes():
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if self.get_burn_time(neighbor) == 0:
                    l1_protection_nodes.add(neighbor)
        return list(l1_protection_nodes)

    def get_l2_protection_nodes(self):
        l2_protection_nodes = set()  # Use a set instead of a list
        for vertex in self.get_l1_protection_nodes():
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if neighbor not in self.get_l1_protection_nodes():
                    l2_protection_nodes.add(neighbor)
        return list(l2_protection_nodes)
    
    def get_n_level_nodes(self, vertex):
        levels = []  # Stores the nodes at each level
        level = 0  # Current level
    
        # Get l0 (level 0)
        levels.append(list(set(self.get_neighbors(vertex)))) 
        finished = False  # Flag to indicate when to stop the loop
        while finished == False:
            level_nodes = []  # Stores the nodes at the current level
            # Iterate over nodes at the current level
            for node in levels[level]:
                # Iterate over neighbors of the current node
                for nbr in set(self.get_neighbors(node)):
                    # Check conditions to add a neighbor to the next level
                    if nbr not in levels[level] and nbr not in levels[level-1] and nbr != vertex:
                        level_nodes.append(nbr)
            levels.append(list(set(level_nodes)))  # Add nodes to the next level
            level += 1  # Move to the next level
            # Check if the current level has no nodes, indicating the end of traversal
            if list(set(level_nodes)) == []:
                finished = True
        return levels

    def get_distance_to_fire(self, vertex):
        if self.is_burning(vertex):
            return 0
        count=1
        #print(vertex)
        for levl in self.get_n_level_nodes(vertex):
            for point in levl:
                if self.is_burning(point):
                    return count
            count+=1
            
    def get_shadow_score(self,working_vertices):
        shadows = [self.get_distance_to_fire(vertex) for vertex in working_vertices]
        sum_shadows = sum(x if x is not None else 0 for x in shadows)
        #if current_period==10:
        #print(shadows)
        return sum_shadows
        
    def is_contaned(self):
        count_open=0
        for burning_vertex in self.get_burning_vertices():
            for neighbor in self.get_neighbors(burning_vertex):
                if not (self.is_protected(neighbor) or self.is_defunct(neighbor) or self.is_burning(neighbor)):
                    count_open+=1
        if count_open==0:
            return True
        else:
            return False
        
    def heuristic(self, current, target):
        # Euclidean distance heuristic
        x1, y1 = current
        x2, y2 = target
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def astar_search(self, start, target):
        open_set = [(0, start)]  # Priority queue for open set
        came_from = {}  # Dictionary to store the parent of each vertex
        g_score = {vertex: float('inf') for vertex in self.vertices}  # Cost from start to each vertex
        g_score[start] = 0
        f_score = {vertex: float('inf') for vertex in self.vertices}  # Estimated total cost from start to goal through each vertex
        f_score[start] = self.heuristic(start, target)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == target:
                # Reconstruct the path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            self.burnt_vertices.add(current)

            for neighbor in self.adjacency_list[current]:
                if neighbor in self.burnt_vertices:
                    continue

                tentative_g_score = g_score[current] + 1  # Distance between current and neighbor is always 1

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, target)
                    if neighbor not in self.protected_vertices and neighbor not in self.defunct_vertices:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found
