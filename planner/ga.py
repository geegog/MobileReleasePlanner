import random

import pandas as pd
import numpy as np
import copy

from planner import base
from planner.lp import LP


class GA(base.MobileReleasePlanner):

    def __init__(self, stakeholder_importance, release_relative_importance, release_duration, coupling=None,
                 crossover_rate=0.1, mutation_rate=0.05):

        # seed: Initial seed solution
        # param m: Population of size
        self.seed = None
        self.metadata = None
        self.m = 50
        # param cr: Crossover Rate
        self.cr = crossover_rate
        # param mr: Mutation Rate
        self.mr = mutation_rate
        self.scored = None

        super(GA, self).__init__(stakeholder_importance, release_relative_importance, release_duration, coupling)
        self.features = self.features()

    def score_population(self):
        """Return a scored and ranked copy of the population.

        This scores the fitness of each member of the population and returns
        the complete population as ``[(solution, score, index)]``.

        Raises:
            Exception: If the population is empty.
        """

        if self.seed is None:
            raise Exception("Cannot score and rank an empty population.")

        scored = [(self.seed[index], self.objective_function(solution), index) for index, solution in
                  enumerate(self.metadata)]
        scored.sort(key=lambda n: n[1])
        scored.reverse()

        return scored

    def proportion_population(self):
        """Return a scored and ranked copy of the population.

        This scores the fitness of each member of the population and returns
        the complete population as `[(member, score, weighted fitness)]`.
        """

        ranked = self.score_population()
        shares = float(sum([t[1] for t in ranked]))

        self.scored = []
        tally = 0
        for tupl in ranked:
            if tupl[1] > 0:
                tally = tally + tupl[1] / shares
            # chromosome, score, share range, index
            self.scored.append((tupl[0], tupl[1], tally, tupl[2]))

    def new_population(self):
        """
        Generate a new population

        :return: A new population of size m
        """
        if self.seed is None:
            self.seed = []
            self.metadata = []
            for _ in range(0, self.m):
                self.seed.append(self.create())

    def create(self):
        """
        The below generates a chromosome
        """

        fx = copy.copy(self.features)
        lp = LP(stakeholder_importance=self.stakeholder_importance,
                release_relative_importance=self.release_relative_importance,
                release_duration=self.release_duration, coupling=self.coupling, highest=False)
        lp.assignment_function(fx)

        sorted_plan = sorted(lp.mobile_release_plan, key=lambda f: f[0])

        solution = self.chromosome(sorted_plan)

        if self.exist(solution):
            self.create()
        else:
            self.metadata.append(sorted_plan)
            return solution

    @staticmethod
    def chromosome(sorted_plan):
        features = []
        for release, was, key, description, effort in sorted_plan:
            features.append(key)
        return features

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
        return self.objective_function(solution)

    def select(self, index=0):
        """
        Get fittest chromosome

        :return: Chooses based on fitness score, a parent for the crossover operation.
        """
        return self.scored[0][index]

    def crossover(self, first_solution, second_solution):
        """
        Performs crossover operation on chromosomes. The crossover operator takes two parents,
        randomly selects items in one parent and fixes their place in the second parent

        :param first_solution: A solution
        :param second_solution: A solution
        :return: Performs crossover on first and second solutions at crossover rate cr.
        """
        size = len(self.keys)
        [fp1, fp2] = random.sample(range(0, size), 2)

        key1 = first_solution[fp1]
        key2 = second_solution[fp2]

        offspring = []
        for index, key in enumerate(second_solution):
            if key == key1 or key == key2:
                offspring.append(key)
            else:
                offspring.append(first_solution[index])

        return offspring

    def mutation(self, solution):
        """
        Performs crossover operation on chromosomes. Random swapping of items in the
        new offspring. The number of swaps is proportional to the mutation rate.

        :param solution: A solution
        :return: Performs mutation on solution at mutation rate mr.
        """
        mutant = list(solution)
        changes = 0
        offset = self.mr

        for locus1 in range(0, len(solution)):
            if random.random() < offset:
                locus2 = locus1
                while locus2 == locus1:
                    locus2 = random.randint(0, len(solution) - 1)

                mutant[locus1], mutant[locus2] = mutant[locus2], mutant[locus1]
                changes += 2
                offset = self.mr / changes

        return mutant

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

    def max(self):
        """
        Get fittest solution in population

        :return: Solution in population P that has the highest fitness score.
        """
        best = self.score_population()[0][0]
        return best


def runner():
    coupling = {("F7", "F8"), ("F9", "F12"), ("F13", "F14")}

    ga = GA(coupling=coupling, stakeholder_importance=(4, 6), release_relative_importance=(0.4, 0.3, 0.3),
            release_duration=14)

    ga.new_population()
    ga.proportion_population()

    print(ga.mobile_release_plan)
    print(ga.objective_function(ga.mobile_release_plan))
    print(ga.effort_release_1, ga.effort_release_2, ga.effort_release_3)


runner()
