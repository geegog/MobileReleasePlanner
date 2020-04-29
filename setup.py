import pandas as pd
import numpy as np
import random
import copy


class LPReleasePlanner(object):

    def __init__(self, stakeholder_importance=(4, 6), release_relative_importance=(0.3, 0.0, 0.7), number_of_releases=3,
                 release_duration=14, effort_release_1=0.0, effort_release_2=0.0, effort_release_3=0.0,
                 coupling=None, precedence=None):
        if precedence is None:
            precedence = {}
        if coupling is None:
            coupling = {}
        self.stakeholder_importance = stakeholder_importance
        self.release_relative_importance = release_relative_importance
        self.number_of_releases = number_of_releases
        self.coupling = coupling
        self.precedence = precedence
        self.effort_release_1 = effort_release_1
        self.effort_release_2 = effort_release_2
        self.effort_release_3 = effort_release_3
        self.release_duration = release_duration
        self.inputs = pd.read_csv("data/sample.csv", skiprows=1, nrows=15,
                                  dtype={"Value value(1,i)": "Int64", "Value value(2,i)": "Int64"})
        self.inputs.columns = ["Feature number", "Feature f(i)", "Effort(days) t(i,2)", "Value v(1,i)",
                               "Urgency u(1,i)", "Value v(2,i)", "Urgency u(2,i)"]
        features = self.inputs["Feature f(i)"].to_xarray().values
        self.results = []
        self.mobile_release_plan = []
        self.delete_flag = False
        self.not_feasible_in_current_mobile_release_plan = []
        self.results.append(features.tolist())

    def calculate_was_for_all_features(self):
        """
        Calculates WAS for all features.

        """
        self.print()
        for k in range(0, self.number_of_releases):
            row = []
            for index, data in self.inputs.iterrows():
                result = self.release_relative_importance[k] * (self.was(
                    [self.get_score(self.stakeholder_importance[0], data[3], self.urgency(vector=data[4], release=k)),
                     self.get_score(self.stakeholder_importance[1], data[5], self.urgency(vector=data[6], release=k))]))
                # (release, WAS, feature_number, feature, effort_estimation)
                row.append((k + 1, result, data[0], data[1], data[2]))
            self.results.append(row)

    @staticmethod
    def was(scores):
        """
        Each value WAS(i, k) is determined as the weighted average of the products of
        the two dimensions of prioritization (stakeholder priorities).

        :param scores: Value of assigning feature(i) to release(r) for each stakeholder
        :return: Weighted Average Satisfaction(was) of assigning feature(i) to release(r).
        """
        was = 0.0
        for value in scores:
            was += value
        return was

    @staticmethod
    def objective_function(was):
        """
        An additive function exists in which the total objective function value is determined
        as the sum of the weighted average satisfaction WAS(i, k) of stakeholder priorities for
        all features f(i) when assigned to release k.

        Note: We consider a solution to be sufficiently good (or qualified) if it achieves
        at least 95 percent of the maximum objective function value.

        :param was: all WAS score of features for release plan (x)
        :return: Objective function score.
        """
        fitness = 0.0
        for _, value, _, _, _ in was:
            fitness += value
        return fitness

    @staticmethod
    def get_score(stakeholder_importance, value_on_feature, urgency_on_feature):
        """
        Stakeholders priorities.

        :param stakeholder_importance: Stakeholders' importance
        :param value_on_feature: Stakeholders' value placed on feature(i)
        :param urgency_on_feature: Stakeholders' urgency placed on feature(i)
        :return: Product of Stakeholders' importance, value, and urgency.
        """
        return stakeholder_importance * value_on_feature * urgency_on_feature

    @staticmethod
    def urgency(vector, release):
        """
        Stakeholders' urgency on a release.

        :param vector: Urgency vector of feature(i)
        :param release: Release
        :return: Urgency placed by stakeholder on release(r).
        """
        vector_tuple = tuple(
            map(lambda v: v, vector.replace("(", "").replace(")", "").replace(" ", "").replace("\'", "").split(",")))
        return int(vector_tuple[release])

    def print(self):
        """
        Print shape and head of input csv file.

        """
        print(self.inputs.shape)
        print(self.inputs.head())

    def assignment_function(self, array_was_feature):
        """
        Greedy function for feature assignment.

        :param array_was_feature: Release and WAS for a feature
        """

        original_feature_set = copy.copy(array_was_feature)
        random.shuffle(array_was_feature)
        for feature_array in array_was_feature:
            if feature_array is not None:
                max_feature = self.get_max_was(feature_array)
                couple_number = self.is_coupled_with(max_feature[2])
                if couple_number is not None:
                    partner = self.get_max_was(original_feature_set[couple_number - 1])
                    total_effort = self.sum_couple_effort(max_feature[4], partner[4])
                    self.assign(max_feature, feature_array, total_effort, partner, original_feature_set[couple_number - 1])
                    if self.delete_flag:
                        index = [idx for idx, f in enumerate(array_was_feature) if (f is not None and f[0][2] == couple_number)]
                        array_was_feature[index[0]] = None
                        self.delete_flag = False
                    continue

                self.assign(max_feature, feature_array)

    def assign(self, max_feature, feature_array, total_effort=None, couple=None, couple_array=None):
        """
        Assigns a feature to a mobile release plan or put in not feasible list if not feasible in current plan.

        :param max_feature: Feature with highest WAS
        :param feature_array: Feature details
        :param total_effort: Total effort estimate of coupled features
        :param couple: Feature partner (couple)
        :param couple_array: Couple details
        """

        if self.can_assign_to_release(self.effort_release_1, max_feature[4], total_effort):
            if couple is not None:
                self.append_to_release(1, max_feature[1], max_feature[2], max_feature[3], max_feature[4], couple)
                self.delete_flag = True
            else:
                self.append_to_release(1, max_feature[1], max_feature[2], max_feature[3], max_feature[4])
        elif self.can_assign_to_release(self.effort_release_2, max_feature[4], total_effort):
            if couple is not None:
                self.append_to_release(2, max_feature[1], max_feature[2], max_feature[3], max_feature[4], couple)
                self.delete_flag = True
            else:
                self.append_to_release(2, max_feature[1], max_feature[2], max_feature[3], max_feature[4])
        elif self.can_assign_to_release(self.effort_release_3, max_feature[4], total_effort):
            if couple is not None:
                self.append_to_release(3, max_feature[1], max_feature[2], max_feature[3], max_feature[4], couple)
                self.delete_flag = True
            else:
                self.append_to_release(3, max_feature[1], max_feature[2], max_feature[3], max_feature[4])
        else:
            self.not_feasible_in_current_mobile_release_plan.append(feature_array)
            if couple_array is not None:
                self.not_feasible_in_current_mobile_release_plan.append(couple_array)
                self.delete_flag = True

    def can_assign_to_release(self, current_total_effort_of_release, effort_estimate, total_effort=None):
        """
        Assigns a feature to a mobile release plan or put in not feasible list if not feasible in current plan.

        :param current_total_effort_of_release: Current sum of effort in a release
        :param total_effort: Total effort estimate of coupled features
        :param effort_estimate: Estimate of a feature

        :return Sum of effort in a release
        """
        if total_effort is None:
            return current_total_effort_of_release + effort_estimate <= self.release_duration
        else:
            return current_total_effort_of_release + total_effort <= self.release_duration

    @staticmethod
    def sum_couple_effort(effort_estimate_1, effort_estimate_2):
        """
        Sum of two estimate values.

        :param effort_estimate_1: First effort estimate
        :param effort_estimate_2: Second effort estimate
        :return: Sum of two estimate values.
        """
        return effort_estimate_1 + effort_estimate_2

    @staticmethod
    def get_max_was(feature_array):
        """
        Selects highest WAS.

        :param feature_array: WAS options
        :return: Highest WAS.
        """
        selection = (0, 0, 0, "", 0)
        for (release, weight, feature_number, feature, effort_estimation) in feature_array:
            if weight > selection[1]:
                selection = (release, weight, feature_number, feature, effort_estimation)
        return selection

    def append_to_release(self, release, weight, feature_number, feature, effort_estimation, couple=None):
        """
        Appends selected feature to mobile release plan.

        :param couple: Coupled feature
        :param release: Release number
        :param weight: WAS
        :param feature_number: feature number
        :param feature: Feature number
        :param effort_estimation: Effort estimate
        """

        if couple is not None:
            self.mobile_release_plan.append((release, couple[1], couple[2], couple[3], couple[4]))
            self.mobile_release_plan.append((release, weight, feature_number, feature, effort_estimation))
            self.increase_effort(release, effort_estimation + couple[4])
        else:
            self.mobile_release_plan.append((release, weight, feature_number, feature, effort_estimation))
            self.increase_effort(release, effort_estimation)

    def increase_effort(self, release, effort_estimation):
        """
        Increase effort of a release.

        :param release: Release number
        :param effort_estimation: Effort estimation
        """

        if release == 1:
            self.effort_release_1 += effort_estimation
        if release == 2:
            self.effort_release_2 += effort_estimation
        if release == 3:
            self.effort_release_3 += effort_estimation

    def is_coupled_with(self, feature_number):
        """
        Get other couple of a feature.

        :param feature_number: Feature number
        :return: Number of the feature's partner.
        """
        coupled_with = None
        for (f1, f2) in self.coupling:
            if feature_number == f1:
                coupled_with = f2
            elif feature_number == f2:
                coupled_with = f1
        return coupled_with


def runner():
    coupling = {(7, 8), (9, 12), (13, 14)}
    precedence = {(2, 1), (5, 6), (3, 11), (8, 9), (13, 15)}

    lp = LPReleasePlanner(coupling=coupling, precedence=precedence)
    lp.calculate_was_for_all_features()

    rows = []
    features = lp.results[0]
    release_1_object_score = lp.results[1]
    release_2_object_score = lp.results[2]
    release_3_object_score = lp.results[3]

    for i in range(0, 15):
        rows.append([release_1_object_score[i], release_2_object_score[i], release_3_object_score[i]])

    # data = np.array(rows)
    # result = pd.DataFrame(data=data)
    # print(result)

    lp.assignment_function(rows)

    print(lp.mobile_release_plan)
    print(lp.objective_function(lp.mobile_release_plan))
    print(lp.effort_release_1, lp.effort_release_2, lp.effort_release_3)


runner()
