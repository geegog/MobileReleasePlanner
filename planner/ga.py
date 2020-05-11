import random

import pandas as pd
import numpy as np
import copy

from planner import base


class GA(base.MobileReleasePlanner):

    def __init__(self, stakeholder_importance, release_relative_importance, release_duration, coupling=None,
                 crossover_rate=0.1, mutation_rate=0.05):

        # seed: Initial seed solution
        # param m: Population of size
        self.seed = None
        self.m = 50
        self.cr = crossover_rate
        self.mr = mutation_rate

        super(GA, self).__init__(stakeholder_importance, release_relative_importance, release_duration, coupling)
        self.features = self.features()

    def new_population(self):
        """
        Generate a new population

        :return: A new population of size m
        """
        if self.seed is None:
            self.seed = []
            for _ in range(0, self.m):
                self.seed.append(self.create())

    def create(self):
        """
        The below generates a chromosome
        """

        solution = copy.copy(self.keys)
        random.shuffle(solution)

        if self.exist(solution):
            self.create()
        else:
            return solution

    def exist(self, solution):
        for s in self.seed:
            if s == solution:
                return True
        return False

    def evaluate(self, solution):
        """
        Get fitness score of chromosome

        :param solution: A chromosome
        :return: Provides a fitness score for a given solution
        """
        pass

    def select(self, population):
        """
        Get fittest chromosome

        :param population: A population
        :return: Chooses based on fitness score, a parent for the crossover operation.
        """
        pass

    def crossover(self, first_solution, second_solution, cr):
        """
        Performs crossover operation on chromosomes

        :param first_solution: A solution
        :param second_solution: A solution
        :param cr: Crossover rate
        :return: Performs crossover on first and second solutions at crossover rate cr.
        """
        pass

    def mutation(self, solution, mr):
        """
        Performs crossover operation on chromosomes

        :param solution: A solution
        :param mr: Mutation rate
        :return: Performs mutation on solution at mutation rate mr.
        """
        pass

    def is_valid(self, solution):
        """
        Check if solution meets constraints

        :param solution: A solution
        :return: Checks validity of solution against the user-defined constraints
        """
        pass

    def back_track(self, solution):
        """
        Proprietary backtracking operation on a given solution.
        This backtracks towards the first parent until a valid
        solution is created or a user-defined number of backtrack operations is reached.

        :param solution: A solution
        :return: A valid solution
        """
        pass

    def cull(self, population):
        """
        Cull(P) removes the (m + 1)th ranked solution from the population, P

        :param population: A solution
        """
        pass

    def check_termination(self):
        """
        A Boolean function which checks if the user’s terminating conditions have been met.
        This may be when a number of optimizations have been completed,
        when there has been no change in the best fitness score over a given number of optimizations,
        a given time has elapsed or the user has interrupted the optimization.

        :return: True or False
        """
        pass

    def max(self, population):
        """
        Get fittest solution in population

        :return: Solution in population P that has the highest fitness score.
        """
        pass


def runner():
    coupling = {("F7", "F8"), ("F9", "F12"), ("F13", "F14")}

    ga = GA(coupling=coupling, stakeholder_importance=(4, 6), release_relative_importance=(0.3, 0.0, 0.7),
            release_duration=14)

    ga.new_population()

    print(ga.mobile_release_plan)
    print(ga.objective_function(ga.mobile_release_plan))
    print(ga.effort_release_1, ga.effort_release_2, ga.effort_release_3)


runner()
