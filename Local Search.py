"""
    The given file is an implementation of the hill climbing algorithm (local search)
    for the Travelling salesman problem.
    """
import re
import numpy as np
from tqdm import tqdm  # Import tqdm for progress visualization
random_seed = 42 #Set a random seed
np.random.seed(random_seed) #use when calling functions using random num generation from numpy


def parse_coordinates_tsp(file_name):
    """
    Function Description: Read in files for different cities and return dictionary with coordinates for each location
    :param filename:
    :type str
    """
    coordinates = {} #Initalize dict to store coordinates output
    coord_section = False
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith("NODE_COORD_SECTION"):
                coord_section = True
                continue
            if coord_section and line.strip() != "EOF":
                parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)  # Match float or integer numbers
                if len(parts) >= 3:
                    city_id = int(parts[0])
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    coordinates[city_id] = (x_coord, y_coord)
    return coordinates


def euclid_distance(city1,city2):
    """
    Calculates the Euclidean distance between two cities represented as coordinates.
    :param city1: Coordinates of city 1
    :param city2: Coordinates of city 2
    :return: Euclidean distance between the cities
    """
    return np.sqrt((city2[0]-city1[0])**2 + (city1[1]-city2[1])**2)


def calculate_paths(coordinates: dict):
    """
    Calculates the distances between all pairs of cities and returns a matrix.
    :param coordinates: Dictionary containing city coordinates
    :return: Matrix of distances between cities
    """
    labels = [i for i in range(1, len(coordinates) + 1)] #List of all of the labels
    paths = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)): #Traverse matrix and add euclidian distance between nodes
        for j in range(len(labels)):
            if i != j:
                paths[i, j] = euclid_distance(coordinates[labels[i]], coordinates[labels[j]])
    return paths

#Initialize solution
#To improve the accuracy of the algorithm I initialize the solution with nearest neighbor algorithm.

def initialize_tour(paths):
    """
    Initializes a tour using the nearest neighbor algorithm.
    :param paths: Matrix of distances between cities
    :return: Initial tour
    """
    number_of_city = paths.shape[0]
    visited_cities = [False] * number_of_city #Intialize the number of visited cities
    tour = [np.random.randint(0, number_of_city)] #Initialize randomized tour
    visited_cities[tour[0]] = True

    while len(tour) < number_of_city:
        closest_city = None #Initialize closest_city
        curr_city = tour[-1]  #Initialize curr city
        nearest_distance = float('inf') #Initialize nearest_distance

        # Find the closest unvisited city relative to the current city
        for city in range(number_of_city):
            if not visited_cities[city] and paths[curr_city, city] < nearest_distance:
                closest_city = city
                nearest_distance = paths[curr_city, city]

        tour.append(closest_city)
        visited_cities[closest_city] = True
    # pass
    return tour

# Function to generate neighboring solutions for the hill climbing algorithm
def generate_neighbors(x, n=10):
    """
    Generates neighboring solutions for the hill climbing algorithm.
    :param x: Current solution (tour)
    :param n: Number of neighbors to generate
    :return: List of generated neighboring solutions
    """
    neighbors = []
    while len(neighbors) < n:
        #Randomly choose the indexes item 1 and 2 to describe subset of curr soln
        item_1 = np.random.randint(0, len(x)-2)
        item_2 = np.random.randint(1, len(x)-1)
        #Inverse the curr soln vertex
        if item_1 > item_2 :
            item_1, item_2 = item_2, item_1
        neighbor = x.copy()
        neighbor[item_1:item_2] = neighbor[item_1:item_2][::-1]
        #Add to the neighbors list
        if tuple(neighbor) not in map(tuple, neighbors):
            neighbors.append(neighbor)
    return neighbors


def fitness(x, paths):
    """
    Calculates the fitness (total distance) of a tour.
    :param x: Tour (list of cities)
    :param paths: Matrix of distances between cities
    :return: Fitness value (total distance of the tour)
    """
    # x is a list of cities
    distance = 0
    for i in range(len(x)-1):
        distance += paths[x[i], x[i+1]]
    distance += paths[x[-1], x[0]]
    # pass
    return distance


def best_neighbor(x: list, paths:np.array, generate_neighbors:callable = generate_neighbors, fitness: callable = fitness):
    """
    Finds the best neighbor among generated neighbors for simple hill climbing.
    :param x: Current solution (tour)
    :param paths: Matrix of distances between cities
    :param generate_neighbors: Function to generate neighbors
    :param fitness: Function to calculate fitness
    :return: Best neighboring solution
    """
    neighbors = generate_neighbors(x)
    best_neighbor = neighbors[0]
    for neighbor in range(1, len(neighbors)):
        if fitness(neighbors[neighbor], paths) < fitness(best_neighbor, paths):
            best_neighbor = neighbors[neighbor]
    # pass
    return best_neighbor


def random_neighbor(x:list, paths:np.array, generate_neighbors:callable = generate_neighbors):
    """
    Generates a random neighboring solution for stochastic hill climbing.
    :param x: Current solution (tour)
    :param paths: Matrix of distances between cities
    :param generate_neighbors: Function to generate neighbors
    :return: Random neighboring solution
    """
    neighbors = generate_neighbors(x)
    # pass
    return neighbors[np.random.randint(0, len(neighbors))]


def hill_climbing(f:callable, x_init:float, n_iters:int, paths:np.array, type:str, epsilon:float = 0.001, steepest:bool = False):
    """
    Performs hill climbing optimization.
    :param f: Fitness function
    :param x_init: Initial solution (tour)
    :param n_iters: Number of iterations
    :param paths: Matrix of distances between cities
    :param type: Type of hill climbing (simple or stochastic)
    :param epsilon: Minimum improvement threshold
    :param steepest: Whether to use steepest ascent/descent
    :return: Optimized solution (tour)
    """
    x = x_init
    x_best = x
    if type == "simple":
        neighbor_function = best_neighbor
    elif type == "stochastic":
        neighbor_function = random_neighbor
    for iter in tqdm(range(n_iters)):
        y = neighbor_function(x, paths)
        if f(x, paths) > f(y, paths) :
            x = y
            if f(x, paths) < f(x_best, paths):
                x_best = x
            else:
                if steepest:
                    x = x_best
    pass
    return x_best


def calculate_cost(tour, paths):
    """
    Calculates the cost (total distance) of a tour.
    :param tour: Tour (list of cities)
    :param paths: Matrix of distances between cities
    :return: Total distance of the tour
    """
    total_cost = 0
    num_cities = len(tour)

    for i in range(num_cities - 1):
        # Accumulate the distance between each pair of consecutive cities in the tour
        total_cost += paths[tour[i], tour[i + 1]]

    # Add the distance from the last city back to the starting city
    total_cost += paths[tour[num_cities - 1], tour[0]]

    return total_cost

# Parse the coordinates from the file
coordinates = parse_coordinates_tsp("Atlanta.tsp")

# Calculate paths between cities
paths = calculate_paths(coordinates)

# Initialize the tour
np.random.seed(42)
initial_tour = initialize_tour(paths)

# Perform hill climbing optimization
np.random.seed(42)
optimized_tour = hill_climbing(fitness, initial_tour, 1000, paths, "simple")

cost_of_tour = calculate_cost(optimized_tour, paths)
print("Cost of optimized tour:", cost_of_tour)
