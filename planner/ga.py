import math
import random

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

from planner import base, crossover
from planner.lp import LP


class GA(base.MobileReleasePlanner):

    def __init__(self, stakeholder_importance, release_relative_importance, release_duration, coupling=None,
                 crossover_rate=0.1, mutation_rate=0.05, max_simulation=600, cross_type='ordered',
                 select_type='fittest', population_size=50):

        self.crossover_type = ["ordered", "partially_matched", "edge_recombination"]
        self.selection_type = ["fittest", "tournament", "proportionate"]
        if cross_type not in self.crossover_type:
            raise TypeError("Value types include: ", self.crossover_type)
        if select_type not in self.selection_type:
            raise TypeError("Value types include: ", self.selection_type)
        self.cross_type = cross_type
        self.select_type = select_type
        # seed: Initial seed solution
        # param m: Population of size
        self.seed = None
        self.m = population_size
        # param cr: Crossover Rate
        self.cr = crossover_rate
        # param mr: Mutation Rate
        self.mr = mutation_rate
        self.scored = None
        self.simulation = 0
        self.max_simulation = max_simulation

        super(GA, self).__init__(stakeholder_importance, release_relative_importance, release_duration, coupling)
        self.features = self.features()

    def ranked(self):
        self.scored.sort(key=lambda n: n[1])
        self.scored.reverse()

        return self.scored

    def score_population(self):
        """Return a scored and ranked copy of the population.

        This scores the fitness of each member of the population and returns
        the complete population as ``[(solution, score, index)]``.

        Raises:
            Exception: If the population is empty.
        """

        if self.seed is None:
            raise Exception("Cannot score and rank an empty population.")

        scored = [(solution, self.evaluate(solution)) for index, solution in
                  enumerate(self.seed)]
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
            # chromosome, score, share range
            self.scored.append((tupl[0], tupl[1], tally))

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

        fx = copy.copy(self.features)
        lp = LP(stakeholder_importance=self.stakeholder_importance,
                release_relative_importance=self.release_relative_importance,
                release_duration=self.release_duration, coupling=self.coupling, highest=False)
        lp.assignment_function(fx)

        sorted_plan = sorted(lp.mobile_release_plan, key=lambda f: f[0])

        solution = self.chromosome(sorted_plan)

        if self.exist(solution):
            return self.create()
        else:
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
        return self.objective_function(self.get_mobile_plan_from_offspring(solution))

    def select_fittest(self, index=0):
        """
        Get fittest chromosome

        :return: Chooses based on fitness score, a parent for the crossover operation.
        """
        return self.scored[index][0]

    def proportionate_select(self):
        """Select a member of the population in a fitness-proportionate way."""
        number = random.random()
        for ticket in self.scored:
            if number < ticket[2]:
                return ticket[0]

        raise Exception("Failed to select a parent. Begin troubleshooting by "
                        "checking your fitness function.")

    def tournament_select(self):
        """Return the best genotype found in a random sample."""
        sample_size = int(math.ceil(self.m * 0.2))
        tournament_size = sample_size

        pop = [random.choice(self.seed)
               for _ in range(0, tournament_size)]

        self.scored = [(geno, self.evaluate(geno)) for geno in pop]
        self.scored.sort(key=lambda n: n[1])
        self.scored.reverse()

        return self.scored[0][0]

    def select(self, index=0):
        if self.select_type == self.selection_type[2]:
            return self.proportionate_select()
        elif self.select_type == self.selection_type[1]:
            return self.tournament_select()
        else:
            return self.select_fittest(index=index)

    def crossover(self, parent1, parent2):
        if self.cross_type == self.crossover_type[1]:
            return crossover.partially_matched(parent1, parent2)[0][0]
        elif self.cross_type == self.crossover_type[2]:
            return crossover.edge_recombination(parent1, parent2)[0]
        else:
            return crossover.ordered(self.keys, parent1, parent2)

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
        validity_check = []
        mrp = self.get_mobile_plan_from_offspring(solution)

        mr1 = [f_tuple for f_tuple in mrp if f_tuple[0] == 1]
        mr2 = [f_tuple for f_tuple in mrp if f_tuple[0] == 2]
        mr3 = [f_tuple for f_tuple in mrp if f_tuple[0] == 3]

        for couple in self.coupling:
            result = self.is_in_release(mr1, couple)
            result2 = self.is_in_release(mr1, (couple[1], couple[0]))
            validity_check.append(result)
            validity_check.append(result2)
        for couple in self.coupling:
            result = self.is_in_release(mr2, couple)
            result2 = self.is_in_release(mr2, (couple[1], couple[0]))
            validity_check.append(result)
            validity_check.append(result2)
        for couple in self.coupling:
            result = self.is_in_release(mr3, couple)
            result2 = self.is_in_release(mr3, (couple[1], couple[0]))
            validity_check.append(result)
            validity_check.append(result2)

        return not (False in validity_check)

    def is_in_release(self, mr, couple):
        for r in mr:
            if r[2] == couple[0]:
                return self.is_feature_in_release(mr, couple[1])
            else:
                continue

    @staticmethod
    def is_feature_in_release(mr, feature):
        is_present = False
        for r in mr:
            if r[2] == feature:
                is_present = True
        return is_present

    def get_mobile_plan_from_offspring(self, solution):
        effort_release_1 = 0.0
        effort_release_2 = 0.0
        effort_release_3 = 0.0
        plan = []

        for key in solution:
            effort = self.effort[self.get_feature_effort_index(key)]
            if effort_release_1 <= self.release_duration and effort_release_1 + effort <= self.release_duration:
                plan.append(self.get_feature_was(1, key))
                effort_release_1 += effort
            elif effort_release_2 <= self.release_duration and effort_release_2 + effort <= self.release_duration:
                plan.append(self.get_feature_was(2, key))
                effort_release_2 += effort
            elif effort_release_3 <= self.release_duration and effort_release_3 + effort <= self.release_duration:
                plan.append(self.get_feature_was(3, key))
                effort_release_3 += effort
            else:
                plan.append(self.get_max_was([f_list for f_list in self.features if f_list[0][2] == key][0],
                                             add_to_release=4))
        return plan

    def get_feature_was(self, release, key):
        return [f_tuple for f_tuple in self.results[release] if f_tuple[2] == key][0]

    def get_feature_effort_index(self, key):
        return self.keys.index(key)

    def ga_operation(self):
        """
        Perform selection, crossover, mutation, and validation.

        :return: A valid solution
        """
        if self.select_type == 'fittest' or self.select_type == 'proportionate':
            self.proportion_population()
        parent1 = self.select()
        parent2 = self.select(index=1)
        if random.random() < self.cr:
            offspring = self.crossover(parent1, parent2)
        else:
            offspring = self.select()
        offspring_from_mr = self.mutation(offspring)
        if self.is_valid(offspring_from_mr):
            return offspring_from_mr
        else:
            return self.ga_operation()

    def cull(self):
        """
        Cull(P) removes the (m + 1)th ranked solution from the population, P

        """
        self.ranked()
        last = self.scored[-1]
        feature = last[0]
        self.seed.remove(feature)

    def check_termination(self):
        """
        A Boolean function which checks if the user’s terminating conditions have been met.
        This may be when a number of optimizations have been completed,
        when there has been no change in the best fitness score over a given number of optimizations,
        a given time has elapsed or the user has interrupted the optimization.

        :return: True or False
        """
        return self.simulation >= self.max_simulation

    def max(self):
        """
        Get fittest solution in population

        :return: Solution in population P that has the highest fitness score.
        """
        best = self.scored[0]
        for f in self.get_mobile_plan_from_offspring(best[0]):
            self.increase_effort(f[0], f[4])
            self.mobile_release_plan.append(f)
        return best

    def solve(self):
        self.new_population()
        terminate_flag = False
        try:
            while not terminate_flag:
                self.simulation += 1
                offspring = self.ga_operation()
                score = self.evaluate(offspring)
                if not self.exist(offspring):
                    self.seed.append(offspring)
                    self.cull()
                terminate_flag = self.check_termination()
        except KeyboardInterrupt:
            pass
        return self.max()

    @staticmethod
    def plot_data(x_axis_data, y_axis_data, x_axis_name, y_axis_name, title):
        plt.style.use('seaborn-whitegrid')

        plt.plot(x_axis_data, y_axis_data)

        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)

        plt.title(title)

        plt.show()


def runner():
    coupling = {("F7", "F8"), ("F9", "F12"), ("F13", "F14")}

    ga = GA(coupling=coupling, stakeholder_importance=(4, 6), release_relative_importance=(0.4, 0.3, 0.3),
            release_duration=14, cross_type='edge_recombination', select_type='fittest', max_simulation=10)

    best = ga.solve()

    print(ga.mobile_release_plan)
    print(ga.objective_function(ga.mobile_release_plan))
    print(ga.effort_release_1, ga.effort_release_2, ga.effort_release_3)


runner()
