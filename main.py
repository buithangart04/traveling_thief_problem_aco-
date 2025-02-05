import numpy as np
import math
from pymoo.indicators.hv import HV
import random
import itertools
import os


class TravelingThiefProblem:
    """
    A class representing for a problem.
    """
    def __init__(self):
        self.num_of_cities = 0
        self.coordinates = []
        self.num_of_items = 0
        self.items = []
        self.R = 0.0
        self.max_weight = 0
        self.min_speed = 0.0
        self.max_speed = 0.0
        self.total_profit = 0.0

    def read_problem(self, file):
        line = file.readline()
        while line:
            if "PROBLEM NAME" in line:
                pass
            elif "KNAPSACK DATA TYPE" in line:
                pass
            elif "DIMENSION" in line:
                self.num_of_cities = int(line.split(":")[1].strip())
                self.coordinates = [[0.0, 0.0] for _ in range(self.num_of_cities)]
            elif "NUMBER OF ITEMS" in line:
                self.num_of_items = int(line.split(":")[1].strip())
                self.items = [0] * self.num_of_items
            elif "RENTING RATIO" in line:
                self.R = float(line.split(":")[1].strip())
            elif "CAPACITY OF KNAPSACK" in line:
                self.max_weight = int(line.split(":")[1].strip())
            elif "MIN SPEED" in line:
                self.min_speed = float(line.split(":")[1].strip())
            elif "MAX SPEED" in line:
                self.max_speed = float(line.split(":")[1].strip())
            elif "EDGE_WEIGHT_TYPE" in line:
                edge_weight_type = line.split(":")[1].strip()
                if edge_weight_type != "CEIL_2D":
                    raise RuntimeError("Only edge weight type of CEIL_2D supported.")
            elif "NODE_COORD_SECTION" in line:
                for i in range(self.num_of_cities):
                    line = file.readline()
                    parts = line.split()
                    self.coordinates[i][0] = float(parts[1].strip())
                    self.coordinates[i][1] = float(parts[2].strip())
            elif "ITEMS SECTION" in line:
                for i in range(self.num_of_items):
                    line = file.readline()
                    parts = line.split()
                    # profit, weight, city
                    profit = float(parts[1].strip())
                    self.items[i] = (profit, float(parts[2].strip()), int(parts[3].strip()) - 1)
                    self.total_profit += profit

            line = file.readline()


class Competition:
    """
    A class representing rules of the competition.
    """
    @staticmethod
    def get_number_of_solutions(problem):
        """
        This function return the number of solutions for each problem
        """
        name = problem.name
        if "a280" in name:
            return 100
        elif "fnl4461" in name:
            return 50
        elif "pla33810" in name:
            return 20
        else:
            return 1000


class NonDominatedSet:
    """
    A class representing a non-dominated set of solutions.
    """

    def __init__(self, number_of_entries):
        # Entries of the non-dominated set
        self.entries = []
        self.number_of_entries = number_of_entries
   
    """
    This function returns the crowding distance score for each solution by calculating the distance of two adjacent solutions
    The meaning of this function is to evaluate the contribution of each solution to the diversity of the non-dominated set.
    """
    def calculate_crowding_distance_solutions(self):
        
        # Get the number of solutions in the list
        num_solutions = len(self.entries)
        solutions = self.entries
        # If there are no solutions, return an empty dictionary
        if num_solutions == 0:
            return {}

        # Get the number of objectives for each solution
        num_objectives = len(solutions[0].objectives)
        # Initialize crowding distances for each solution to 0
        crowding_distances = {sol: 0 for sol in solutions}

        # Iterate over each objective
        for m in range(num_objectives):
            # Sort solutions based on the current objective values
            sorted_solutions = sorted(solutions, key=lambda sol: sol.objectives[m])

            # Assign infinite crowding distance to boundary solutions
            crowding_distances[sorted_solutions[0]] = float('inf')
            crowding_distances[sorted_solutions[-1]] = float('inf')
            
            # Get the minimum and maximum values for the current objective
            min_value = sorted_solutions[0].objectives[m]
            max_value = sorted_solutions[-1].objectives[m]
            
            # If all solutions have the same objective value, skip this objective
            if max_value - min_value == 0:
                continue
            
            # Calculate crowding distances for intermediate solutions
            for i in range(1, num_solutions - 1):
                # Objective values of the next and previous solutions
                next_obj = sorted_solutions[i + 1].objectives[m]
                prev_obj = sorted_solutions[i - 1].objectives[m]
                
                # Add the normalized difference to the crowding distance
                crowding_distances[sorted_solutions[i]] += (next_obj - prev_obj) / (max_value - min_value)

        return crowding_distances

    def remove_lowest_crowding_distance(self):
        # Calculate crowding distances for all solutions
        crowding_distances = self.calculate_crowding_distance_solutions()
        
        # Initialize variables to track the solution with the lowest crowding distance
        min_distance = float('inf') # Set the initial minimum distance to infinity
        solution_to_remove = None # Variable to hold the solution to be removed
        
        # Iterate through each solution and its corresponding crowding distance
        for sol, distance in crowding_distances.items():
            # Update the solution with the lowest crowding distance
            if distance < min_distance:
                min_distance = distance
                solution_to_remove = sol
        
        # If a solution with the lowest crowding distance is found, remove it from the entries
        if solution_to_remove is not None:
            self.entries.remove(solution_to_remove)

        return solution_to_remove

    def add(self, solution):
        """
        Add a solution to the non-dominated set.
        
        :param solution: The solution to be added.
        :return: True if the solution was added; otherwise, False.
        """
        is_added = True

        # Use a copy of the list to avoid modifying it while iterating
        for other in self.entries[:]:
            rel = solution.get_relation(other)

            # If this solution is dominated or equal in design space
            if rel == -1 or (rel == 0 and solution.equals_in_design_space(other)):
                is_added = False
                break
            
            # If this solution dominates other solution, remove the dominated one
            elif rel == 1:
                self.entries.remove(other)
        
        # append this solution to the set
        if is_added:
            self.entries.append(solution)
        
        # check solution, if it is added to the list and the length of the list is over the number of solutions
        # , find and remove the solution that has the lowest crowding distance score.'''
        if is_added and len(self.entries) > self.number_of_entries:
            self.remove_lowest_crowding_distance()

        return is_added


class Solution:
    """
        This is a solution objective class that stores the tour, packing plan,
        and objective values.
    """

    def __init__(self, time, profit, tour, plan):
        # The tour of the thief
        self.pi = tour

        # The packing plan
        self.z = plan

        # The time the thief needed for traveling
        self.time = time

        # The profit the thief made on that tour
        self.profit = -profit

        # Objective value for solving single-objective problems using R
        self.single_objective = -1.0

        # The objective values of the function
        self.objectives = (time, -profit)
    """
        this function is used for non-dominated sorting and returns the dominance relation.

        :param other: Solution to compare with.
        :return: 1 if dominates, -1 if dominated, 0 if indifferent.
    """
    def get_relation(self, other):
        val = 0
        for i in range(len(self.objectives)):
            if self.objectives[i] < other.objectives[i]:
                if val == -1:
                    return 0
                val = 1
            elif self.objectives[i] > other.objectives[i]:
                if val == 1:
                    return 0
                val = -1

        return val
    """
        Compare if the tour and packing plan are equal.

        :param other: Solution to compare with.
        :return: True if tour and packing plan are equal, False otherwise.
        """
    def equals_in_design_space(self, other):
        return self.pi == other.pi and np.array_equal(self.z, other.z)


class ACO:
    def __init__(self, problem, num_of_solutions, Q, _rho, _alpha, _beta, iterations, n_ants):
        self.nds = NonDominatedSet(num_of_solutions) # solution archive
        self.Q = Q # fitness coefficent
        self._rho = _rho # pheromone evaporation rate
        self._alpha = _alpha # the parameter to control pheromone influence 
        self._beta = _beta # the parameter to control local heuristic influence 
        self.iterations = iterations # number of iterations
        self.n_ants = n_ants # number of ants for each iteration
        
        # random pheromone matrix and fill the matrix diagonal with value 0
        self.pheromone_matrix = np.random.uniform(0.1, 1, (problem.num_of_cities, problem.num_of_cities))
        np.fill_diagonal(self.pheromone_matrix, 0)
        
        self.problem = problem

    def swap_cities(self, tour, n):
        # if the number of cites is less than 2, we can not swap 2 different cities because the start city always is city 0
        if n <= 2:
            return tour

        # swap the second and the last city(except the first city, the tour like 0,2,1,...,5,0. so we don't take the
        # first and the last elements)
        i = random.randint(1, n - 1)
        j = random.randint(1, n - 1)
        
        
        # keep random until we get the different indexs
        while i == j:
            j = random.randint(1, n - 1)
        new_tour = tour.copy()
        
        # swap two random cites
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        
        return new_tour

    def construct_tsp_tour(self, num_of_cities):
        # Initialize the tour starting at city 0
        tour = [0]
        
        # Create a list to track visited cities, marking the starting city as visited
        visited = [False for _ in range(num_of_cities)]
        visited[0] = True
        
        # Continue constructing the tour until all cities are visited
        while len(tour) < num_of_cities:
            # Get the current city (last city in the tour)
            current_city = tour[-1]
        
            # Initialize a list to store probabilities for selecting the next city
            probabilities = []
            # Iterate over all cities to calculate their probabilities
            for j in range(num_of_cities):
                if not visited[j]:
                    # Retrieve the pheromone value for the path from the current city to city j
                    tau = self.pheromone_matrix[current_city][j]
                    
                    # Ensure pheromone levels stay within the range [0.1, 1] to prevent premature convergence
                    if tau < 0.1:
                        self.pheromone_matrix[current_city][j] = 0.1
                        tau = 0.1
                    elif tau > 1:
                        self.pheromone_matrix[current_city][j] = 1
                        tau = 1
                    
                    # Calculate the heuristic value (eta) based on the distance between cities
                    distance = np.linalg.norm(np.array(self.problem.coordinates[current_city]) - np.array(self.problem.coordinates[j]))
                    eta = 1.0/distance if distance != 0 else 1.0
                    
                    # Calculate the probability for city j and append it
                    probabilities.append((tau ** self._alpha) * (eta ** self._beta))
                else:
                    # Assign a probability of 0 for already visited cities
                    probabilities.append(0)
            
            # Convert the probabilities list to a NumPy array
            probabilities = np.array(probabilities)
            
            # Normalize the probabilities to ensure they sum to 1
            probabilities /= probabilities.sum()
            
            # Select the next city based on the calculated probabilities
            next_city = np.random.choice(range(num_of_cities), p=probabilities)
            tour.append(next_city)
            visited[next_city] = True
        
        # Complete the tour by returning to the starting city
        tour.append(0)
        
        return tour

    def construct_packing_plan(self, tour, n_tries):
        best_plan = [{}]
        best_profit = [0, 0]
        selected_bags = []

        # Precompute the distance from each city to the end of the tour
        distance_until_end = [0 for _ in range(self.problem.num_of_cities)]
        for i in range(len(tour) - 2, 0, -1):
            distance_until_end[tour[i]] = distance_until_end[tour[i + 1]] + np.linalg.norm(np.array(self.problem.coordinates[tour[i]]) - np.array(self.problem.coordinates[tour[i + 1]]))

        for _ in range(n_tries):
            # Generate random weights (theta, delta, gamma) for scoring items, and ensure that theta + delta + gamma = 1
            theta, delta, gamma = np.random.rand(3)
            norm = theta + delta + gamma
            theta, delta, gamma = theta / norm, delta / norm, gamma / norm

            scores = []
            # Calculate scores for all items based on profit, weight, and distance 
            for index, value in enumerate(problem.items):
                profit, weight, city = value
                if weight == 0 or distance_until_end[city] == 0:
                    score = 1 # Avoid division by zero
                else:
                    score = (profit ** theta) / ((weight ** delta) * (distance_until_end[city] ** gamma))
                scores.append((score, profit, weight, city, index))
            
            # Sort items by their scores in descending order
            scores.sort(reverse=True, key=lambda x: x[0])
            
            current_weight = 0 # Current weight of the packing plan
            total_profit = [0] # List of total profits for each plan
            current_plan = [{}] # Packing plans for each attempt
            current_selected_bags = [{}] # Selected bags for each plan
            visited = [False for _ in range(self.problem.num_of_cities)] # Track visited cities

            # Handle large-scale problems with a different strategy for efficiency
            if problem.num_of_items > 30000:
                random_num_of_items = 0
                count = 0
                for _, profit, weight, city, index in scores:
                    if current_weight + weight <= problem.max_weight:
                        # Switch to a new plan after adding a random number of items
                        if count == random_num_of_items:
                            # get the previous plan and append this bag to this plan
                            previous_profit = total_profit[-1]
                            previous_plan = current_plan[-1].copy()
                            previous_selected_bags = current_selected_bags[-1].copy()
                            random_num_of_items = random.randint(1, 400)
                            count = 0

                        if not visited[city]:
                            previous_plan[city] = 0
                            visited[city] = True
                        previous_plan[city] += weight
                        current_weight += weight
                        previous_profit += profit
                        previous_selected_bags[index] = 1
                        count += 1
                        
                        # Save the new plan
                        if count == random_num_of_items:
                            total_profit.append(previous_profit)
                            current_plan.append(previous_plan)
                            current_selected_bags.append(previous_selected_bags)
                    else:
                        break
                    
                    # Update the best plan and profit if this plan is better
                    if len(total_profit) > 1 and total_profit[1] > best_profit[1]:
                        best_plan = current_plan
                        best_profit = total_profit
                        selected_bags = current_selected_bags
            else:
                # Handle smaller problems by adding one item at a time
                number_of_selected_bags = 1
                for _, profit, weight, city, index in scores:
                    if current_weight + weight <= problem.max_weight:
                        # Build the plan incrementally by adding one item
                        previous_profit = total_profit[number_of_selected_bags - 1]
                        previous_plan = current_plan[number_of_selected_bags - 1].copy()
                        previous_selected_bags = current_selected_bags[number_of_selected_bags - 1].copy()
                        
                        if not visited[city]:
                            previous_plan[city] = 0
                            visited[city] = True
                        previous_plan[city] += weight
                        current_weight += weight
                        previous_profit += profit
                        previous_selected_bags[index] = 1
                        
                        # Save the updated plan
                        total_profit.append(previous_profit)
                        current_plan.append(previous_plan)
                        current_selected_bags.append(previous_selected_bags)
                        number_of_selected_bags += 1
                    else:
                        break
                    
                    # Update the best plan and profit if this plan is better
                    if total_profit[1] > best_profit[1]:
                        best_plan = current_plan
                        best_profit = total_profit
                        selected_bags = current_selected_bags

        return selected_bags, best_plan, best_profit

    def calculate_total_time(self, tour, best_plan, capacity, max_speed, min_speed):
        # Initialize the total time spent on the tour
        total_time = 0
        # Initialize the current weight of the items carried by the traveler
        current_weight = 0
        
        # Iterate through each pair of consecutive cities in the tour
        for i in range(len(tour) - 1):
            city = tour[i]
            next_city = tour[i + 1]
            
            # Update the current weight by adding the weight of items picked up in the current city
            current_weight += best_plan.get(city, 0)
            
            # Calculate the current speed of travel based on the weight carried
            # Speed decreases linearly as the weight approaches the capacity
            current_speed = max_speed - current_weight * (max_speed - min_speed) / capacity
            
            # Ensure that the speed does not drop below the minimum speed
            if current_speed < min_speed: current_speed = min_speed
            
            # distance = self.distance_matrix[city][next_city]
            distance = np.linalg.norm(np.array(self.problem.coordinates[city]) - np.array(self.problem.coordinates[next_city]))
            
            # Add the time taken to travel between the two cities to the total time
            total_time += distance / current_speed

        return total_time
    
    """
        This function calculates the amount of pheromones updated aftertraveling each tour
    """
    def calculate_fitness(self, total_profit, total_time):
        return self.Q * total_profit / total_time
    
    """
        This function updates pheromones after each tour
    """
    def update_pheromones(self, tour, fitness):
        self.pheromone_matrix *= (1 - self._rho)
        for i in range(len(tour) - 1):
            self.pheromone_matrix[tour[i]][tour[i + 1]] += fitness

    def update_solutions_by_tour(self, tour, is_original_tour):
        selected_bags, best_plan, total_profit = self.construct_packing_plan(tour, 3)
        for i in range(len(total_profit)):
            total_time = self.calculate_total_time(tour, best_plan[i],
                                                   self.problem.max_weight, self.problem.max_speed,
                                                   self.problem.min_speed)
            
            # Generate a plan that represents whether each item is selected or not.
            # Each element corresponds to an item, with 1 indicating the item is selected and 0 otherwise.
            plan = [selected_bags[i].get(j, 0) for j in range(len(self.problem.items))]
            
            # Create a solution object with the tour and plan 
            sol = Solution(total_time, total_profit[i], tour, plan)
            
            # Add the solution to the non-dominated set (NDS).
            self.nds.add(sol)
            
            # If the current tour is the original tour, update pheromones after this tour
            if is_original_tour:
                fitness = self.calculate_fitness(total_profit[i], total_time)
                self.update_pheromones(tour, fitness)

    def solve(self):
        for _ in range(self.iterations):
            print(f"Iteration: {_}")
            for _ in range(self.n_ants):
                tour = self.construct_tsp_tour(self.problem.num_of_cities)
                
                # apply local search to avoid local mimimum value
                new_tour = self.swap_cities(tour, self.problem.num_of_cities)
                self.update_solutions_by_tour(tour, True)
                self.update_solutions_by_tour(new_tour, False)
        
        return self.nds

class Utils:
    def __init__(self):
        pass

    def write_2_file(self, nds, vars, instance_name):
        # Define the base folder for saving results
        base_folder = "results"
        
        # Create a subfolder specific to the current optimization instance
        sub_folder = instance_name + '/optimization'
        folder_path = os.path.join(base_folder, sub_folder)
        
        # Ensure the folder path exists, create if not
        os.makedirs(folder_path, exist_ok=True)
        prefix = f'Group16_{instance_name}_optimize'
        
        # write the final solutions
        f_file_path = os.path.join(folder_path, f'{prefix}.f')
        
        # Extract objective values from all solutions
        objectives = np.array([sol.objectives for sol in nds.entries])
        
        with open(f_file_path, "a") as file:
            file.write(vars + '\n')
            
            # Write each solution's objective values to the file
            for row in objectives:
                row_copy = row.copy()
            
                # Convert the last objective to its absolute value
                row_copy[-1] = abs(row_copy[-1])
                
                # Format the objective values as space-separated numbers
                formatted_row = " ".join(map(str, row_copy))
                file.write(formatted_row + "\n")
        
        # Write the tours and plans associated with the solutions to another file
        x_file_path = os.path.join(folder_path, f'{prefix}.x')
        with open(x_file_path, "a") as file:
            file.write(vars + '\n')
            
            # Write details of each solution (tour and plan) to the file
            for sol in nds.entries:
                tour = sol.pi
                plan = sol.z
                tour_info = " ".join(map(str, [x + 1 for x in tour[:len(tour) - 1]]))
                plan_info = " ".join(map(str, plan))
                file.write(tour_info + "\n" + plan_info + "\n\n")
        
        return (f_file_path, x_file_path)
    
    """
        This function calculates hv after running optimization of all combinations of hyperparameters
        , and write the results which have the best hv
    """
    def calculate_hv(self, instance_name, Z_ideal, Z_nadir, f_file_path, x_file_path):
        # Normalize the objectives based on ideal and nadir values from the problem
        z_ideal = np.array(Z_ideal[instance_name])
        z_nadir = np.array(Z_nadir[instance_name])
        sols = {} # Dictionary to store solutions
        with open(f_file_path, 'r') as file:
            line = file.readline()
            this_key = None # Variable to keep track of the current solution key
            while line:
                line = line.strip() # Strip whitespace from the line
                if line:
                    if line.startswith('--'): # If the line starts with '--', it's a new solution key
                        this_key = line
                        sols[this_key] = []
                    else:
                        row = [float(x) for x in line.split()]
                        
                        # get negative value of profit to calculate hv
                        row[1] = -row[1]
                        
                        # Update the z_ideal and z_nadir values based on the solution to maintain the correct bounds
                        for i in range(len(row)):
                            if row[i] > z_nadir[i]:
                                z_nadir[i] = row[i]
                            if row[i] < z_ideal[i]:
                                z_ideal[i] = row[i]
                        
                        # Append the solution to the list under the current key
                        sols[this_key].append(row) 
                line = file.readline()
        
        # Initialize variables to track the best HV and corresponding solution
        best_hv_key = None
        best_hv = 0
        best_objectives = None
        
        # Initialize the Hypervolume object with a reference point (1, 1)
        hv = HV(ref_point=(1, 1))
        prefix = f'Group16_{instance_name}'
        for key, value in sols.items():
            objectives = np.array(value)
            
            # Normalize the objectives by scaling them based on z_ideal and z_nadir
            for i in range(objectives.shape[0]):
                objectives[i][0] = (objectives[i][0] - z_ideal[0]) / (z_nadir[0] - z_ideal[0])
                objectives[i][1] = (objectives[i][1] - z_ideal[1]) / (z_nadir[1] - z_ideal[1])
            
            # Calculate the Hypervolume for the current solution set
            hv_value = hv(objectives)
            with open(f'results/{instance}/optimization/{prefix}_optimize.hv', 'a') as file:
                file.write(key + '\n')
                file.write(str(hv_value) + '\n')
            
            # If this solution has the best HV so far, update the best HV and the corresponding solution
            if hv_value > best_hv:
                best_hv = hv_value
                best_hv_key = key
                best_objectives = value
        
        # Write the final best HV and z_ideal, z_nadir values to a file
        with open(f'results/{instance}/{prefix}.hv', 'a') as file:
            # file.write(best_hv_key + '\n')
            file.write(str(best_hv) + '\n')
            file.write(f'z_ideal: {z_ideal}, z_nadir: {z_nadir}' + '\n')
        
        # Write the best solution's objectives to a file
        with open(f'results/{instance}/{prefix}.f', 'a') as file:
            # file.write(best_hv_key + '\n')
            for row in best_objectives:
                row_copy = row.copy()
                row_copy[-1] = abs(row_copy[-1])
                formatted_row = " ".join(map(str, row_copy))
                file.write(formatted_row + "\n")
        
        # Write the tour and plan associated with the best solution to a file
        with open(f'results/{instance}/{prefix}.x', 'a') as file:
            with open(x_file_path, 'r') as x_file:
                line = x_file.readline()
                is_append = False
                while line:
                    if is_append:
                        if line.startswith('--'):
                            break
                        file.write(line)
                    if best_hv_key in line:
                        # file.write(best_hv_key + '\n')
                        is_append = True
                    line = x_file.readline()


if __name__ == "__main__":
    instance_2_run = ['a280-n279', 'a280-n1395', 'a280-n2790', 'fnl4461-n4460', 'fnl4461-n22300', 'fnl4461-n44600',
                      'pla33810-n33809', 'pla33810-n169045', 'pla33810-n338090']
    Z_ideal = {'test-example-n4': [20.0, -74], 'a280-n279': [2613.0, -42036.0], 'a280-n1395': [2613.0, -489194.0],
               'a280-n2790': [2613.0, -1375443.0],
               'fnl4461-n4460': [185359.0, -645150.0], 'fnl4461-n22300': [185359.0, -7827881.0],
               'fnl4461-n44600': [185359.0, -22136989.0]
        , 'pla33810-n33809': [66048945.0, -4860715.0], 'pla33810-n169045': [66048945.0, -59472432.0],
               'pla33810-n338090': [66048945.0, -168033267.0]}
    Z_nadir = {'test-example-n4': [38.91, -0.0], 'a280-n279': [5444.0, -0.0], 'a280-n1395': [6573.0, -0.0],
               'a280-n2790': [6646.0, -0.0],
               'fnl4461-n4460': [442464.0, -0.0], 'fnl4461-n22300': [452454.0, -0.0],
               'fnl4461-n44600': [459901.0, -0.],
               'pla33810-n33809': [168432301.0, -0.0], 'pla33810-n169045': [169415148.0, -0.0],
               'pla33810-n338090': [168699977.0, -0.0]}

    for instance in instance_2_run:
        with open(f'resources/{instance}.txt', 'r', encoding='utf-8') as file:
            problem = TravelingThiefProblem()
            problem.read_problem(file)
            problem.name = instance
            num_of_solutions = Competition.get_number_of_solutions(problem)
        
        ant_counts = [5] # number of ants in each iteration
        alphas = [2] # the parameter to control pheromone influence 
        betas = [2] # the parameter to control local heuristic influence 
        evaporation_rates = [0.1]
        fitness_coefficients = [0.4]
        iteration_counts = [500] # number of iterations 
        # loop through all combinations of hyperparameters
        params = itertools.product(ant_counts, alphas, betas, evaporation_rates, fitness_coefficients, iteration_counts)

        util = Utils()

        f_file_path = None
        x_file_path = None

        # Solve the problem with the current parameters
        for no_ants, alpha, beta, rho, Q, iterations in params:
            aco = ACO(problem, num_of_solutions, Q, rho, alpha, beta, iterations, no_ants)
            nds = aco.solve()
            vars = f'----------no_ants : {no_ants},alpha: {alpha}, beta: {beta}, rho: {rho}, Q: {Q},iterations: {iterations}----------'
            
            # write the solutions to file
            f_file_path, x_file_path = util.write_2_file(nds, vars, instance)
        
        # to find optimized parameters, we compare the hv between the solutions, and we have to calculate it finally
        # because we need to have the same ideal and nadir points for all parameter combinations
        util.calculate_hv(instance, Z_ideal, Z_nadir, f_file_path, x_file_path)
