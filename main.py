import numpy as np
import math
from pymoo.indicators.hv import HV

class TravelingThiefProblem:
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
    @staticmethod
    def number_of_solutions(problem):
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
    
    import numpy as np

    def calculate_crowding_distance_solutions(self):
        num_solutions = len(self.entries)
        solutions = self.entries
        if num_solutions == 0:
            return {}

        num_objectives = len(solutions[0].objectives)
        crowding_distances = {sol: 0 for sol in solutions}

        for m in range(num_objectives):
            sorted_solutions = sorted(solutions, key=lambda sol: sol.objectives[m])

            crowding_distances[sorted_solutions[0]] = float('inf')
            crowding_distances[sorted_solutions[-1]] = float('inf')

            min_value = sorted_solutions[0].objectives[m]
            max_value = sorted_solutions[-1].objectives[m]
            
            if max_value - min_value == 0:
                continue

            for i in range(1, num_solutions - 1):
                next_obj = sorted_solutions[i + 1].objectives[m]
                prev_obj = sorted_solutions[i - 1].objectives[m]
                crowding_distances[sorted_solutions[i]] += (next_obj - prev_obj) / (max_value - min_value)
        
        return crowding_distances
    
    def remove_lowest_crowding_distance(self):
        crowding_distances = self.calculate_crowding_distance_solutions()

        min_distance = float('inf')
        solution_to_remove = None

        for sol, distance in crowding_distances.items():
            if distance < min_distance:
                min_distance = distance
                solution_to_remove = sol

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

            # If dominated by or equal in design space
            if rel == -1 or (rel == 0 and solution.equals_in_design_space(other)):
                is_added = False
                break
            elif rel == 1:
                self.entries.remove(other)

        if is_added:
            self.entries.append(solution)
        
        if is_added & len(self.entries) > self.number_of_entries:
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

    def get_relation(self, other):
        """
        Used for non-dominated sorting and returns the dominance relation.

        :param other: Solution to compare with.
        :return: 1 if dominates, -1 if dominated, 0 if indifferent.
        """
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

    def equals_in_design_space(self, other):
        """
        Compare if the tour and packing plan are equal.

        :param other: Solution to compare with.
        :return: True if tour and packing plan are equal, False otherwise.
        """
        return self.pi == other.pi and self.z == other.z

class ACO:

    def calculate_distance_matrix(self, coordinates):
        return [[ math.dist(p1,p2) if i != j else 0 
                for j, p1 in enumerate(coordinates)] 
                for i, p2 in enumerate(coordinates)]

    """calculate local heuristic in the constructing tour part"""
    def calculate_local_heuristic(self, coordinates):
        return [[1 / math.dist(p1, p2) if i != j and math.dist(p1, p2) != 0  else 0 
             for j, p1 in enumerate(coordinates)] 
            for i, p2 in enumerate(coordinates)]
    
    def swap_cities(self, tour, i, j):
        new_tour = tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def construct_tsp_tour(self, num_of_cities, pheromone_matrix, local_heuristic, _alpha, _beta):
        tour = [0]
        while len(tour) < num_of_cities:
            current_city = tour[-1]
            probabilities = []
            for j in range(num_of_cities):
                if j not in tour:
                    tau = pheromone_matrix[current_city][j]
                    eta = local_heuristic[current_city] [j]
                    probabilities.append((tau ** _alpha) * (eta ** _beta))
                else:
                    probabilities.append(0)
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            next_city = np.random.choice(range(num_of_cities), p=probabilities)
            tour.append(next_city)
        # the tour start and end at the first city
        tour.append(0)
        return tour

    def construct_packing_plan(self, problem, tour, ptries, distance_matrix):
        best_plan = [{}]
        selected_bags = [[]]
        best_profit = [0, 0]
        
        for _ in range(ptries):
            # random theta, delta, gamma between 0 and 1, and theta + delta + gamma = 1 
            theta, delta, gamma = np.random.rand(3)
            norm = theta + delta + gamma
            theta, delta, gamma = theta / norm, delta / norm, gamma / norm
            
            scores = []
            for index, value in enumerate(problem.items):
                profit, weight, city = value
                distance = sum(distance_matrix[city][c] for c in tour if c >= city)
                if weight == 0 or distance == 0:
                    score = 0
                else:
                    score = (profit ** theta) / ((weight ** delta) * (distance ** gamma))
                scores.append((score, profit, weight, city, index))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            current_weight = 0
            # list of plans, the first plan always select no bag.
            total_profit = [0]
            current_plan = [{}]
            current_selected_bags = [[]]
            #start with 1 selected bag
            number_of_selected_bags = 1
            for _, profit, weight, city, index in scores:
                if current_weight + weight <= problem.max_weight:
                    # get the previous plan and append this bag to this plan
                    previous_profit = total_profit[number_of_selected_bags -1]
                    previous_selected_bags = current_selected_bags[number_of_selected_bags -1].copy()
                    previous_plan = current_plan[number_of_selected_bags -1].copy()
                    if city not in previous_plan:
                        previous_plan[city] = 0
                    previous_plan[city] += weight
                    previous_selected_bags.append(index)
                    current_weight += weight
                    previous_profit += profit
                    
                    total_profit.append(previous_profit)
                    current_plan.append(previous_plan)
                    current_selected_bags.append(previous_selected_bags)
                    number_of_selected_bags += 1
                else : break
            if total_profit[1] > best_profit[1]:
                best_plan = current_plan
                selected_bags = current_selected_bags 
                best_profit = total_profit
        
        return (selected_bags, best_plan, best_profit)

    def calculate_total_time (self, distance_matrix, tour, best_plan, capacity, max_speed, min_speed):
        total_time = 0
        current_weight = 0
        for i in range(len(tour) - 1):
            city = tour[i]
            next_city = tour[i + 1]
            current_weight += best_plan.get(city, 0)
            current_speed = max_speed - current_weight/capacity
            if current_speed < min_speed : current_speed = min_speed
            distance = distance_matrix[city][next_city]
            total_time += distance / current_speed

        return total_time
    
    def calculate_fitness(self, Q, total_profit, total_time):
        return Q * total_profit / total_time
    
    def update_pheromones(self, pheromone_matrix, _rho, tour, fitness):
        pheromone_matrix *= (1 - _rho)
        for i in range(len(tour) - 1):
            pheromone_matrix[tour[i]][tour[i+1]] += fitness

    def update_solutions_by_tour(self, tour, nds, problem, is_original_tour, distance_matrix, pheromone_matrix, _rho, Q):
        selected_bags, best_plan, total_profit = self.construct_packing_plan(problem, tour, 50, distance_matrix)
        for i in range(len(total_profit)):
            total_time = self.calculate_total_time (distance_matrix, tour, best_plan[i],
                                                        problem.max_weight, problem.max_speed, problem.min_speed)
            sol = Solution(total_time, total_profit[i], tour, selected_bags[i])
            nds.add(sol)
            if is_original_tour == True:
                fitness = self.calculate_fitness(Q, total_profit[i], total_time)
                # print(fitness)
                self.update_pheromones(pheromone_matrix, _rho, tour, fitness)
        

    def solve(self, problem, num_of_solutions):
        nds = NonDominatedSet(num_of_solutions)
        # number of population
        lst_no_ants = [10]
        # pheromone coefficent
        lst_alpha = [1]
        # heuristic coefficent
        lst_beta = [1]
        #evaporation rate
        lst_evaporation_rate = [0.3]
        # fitness coefficent
        lst_fitness_coefficient = [0.5]
        # number of generations
        iterations = 10
        # pheromone matrix
        pheromone_matrix = np.random.uniform(0.1, 1, (problem.num_of_cities, problem.num_of_cities))
        np.fill_diagonal(pheromone_matrix, 0)
        # calculate distance matrix of all edges
        distance_matrix = self.calculate_distance_matrix(problem.coordinates)
        local_heuristic = self.calculate_local_heuristic(problem.coordinates)
        for Q in lst_fitness_coefficient:
            for _rho in lst_evaporation_rate:
                for _alpha in lst_alpha:
                    for _beta in lst_beta:
                        for _ in range(iterations):
                            for no_ants in lst_no_ants:
                                tour = self.construct_tsp_tour(problem.num_of_cities, pheromone_matrix, local_heuristic, _alpha, _beta)
                                # apply local search to avoid local mimimum value
                                # swap the first and the last city(except the first city, the tour like 0,2,1,...,5,0. so we don't take the first and the last elements)
                                new_tour = self.swap_cities(tour, 1, problem.num_of_cities - 1)
                                # print(new_tour)
                                self.update_solutions_by_tour(tour, nds, problem, True, distance_matrix, pheromone_matrix, _rho, Q)
                                self.update_solutions_by_tour(new_tour, nds, problem, False, distance_matrix, pheromone_matrix, _rho, Q)
        
        return nds
                                
    

def main():
    instance_2_run = ['a280-n279']
    for instance in instance_2_run:
        with open(f'resources/{instance}.txt', 'r', encoding='utf-8') as file:
            problem = TravelingThiefProblem()
            problem.read_problem(file)
            problem.name = instance
            num_of_solutions = Competition.number_of_solutions(problem)
            aco= ACO()
            nds = aco.solve(problem, num_of_solutions)
            ref_point = (66646.0, -0.0)
            objectives = np.array([sol.objectives for sol in nds.entries])
            print(objectives)
            hv = HV(ref_point=ref_point)
            print(f'HV: {hv(objectives)}')
            

if __name__ == "__main__":
    main()


