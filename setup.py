import pandas as pd
import numpy as np


class LPReleasePlanner(object):

    def __init__(self, stakeholder_importance=(4, 6), release_relative_importance=(0.7, 0.3, 0.0), number_of_releases=3,
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
        for value in was:
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
        Assigns a feature to a mobile release plan.

        :param array_was_feature: Release and WAS for a feature
        """

        for feature_array in array_was_feature:
            max_feature = self.get_max_was(feature_array)
            if self.effort_release_1 + max_feature[4] <= self.release_duration:
                self.append_to_release(1, max_feature[1], max_feature[2], max_feature[3], max_feature[4])
            elif self.effort_release_2 + max_feature[4] <= self.release_duration:
                self.append_to_release(2, max_feature[1], max_feature[2], max_feature[3], max_feature[4])
            elif self.effort_release_3 + max_feature[4] <= self.release_duration:
                self.append_to_release(3, max_feature[1], max_feature[2], max_feature[3], max_feature[4])
            else:
                self.not_feasible_in_current_mobile_release_plan.append(feature_array)

    @staticmethod
    def get_max_was(feature_array):
        """
        Selects highest WAS.

        :param feature_array: WAS options
        :return:  Highest WAS.
        """
        selection = (0, 0, 0, "", 0)
        del feature_array[0:1]
        for (release, weight, feature_number, feature, effort_estimation) in feature_array:
            if weight > selection[1]:
                selection = (release, weight, feature_number, feature, effort_estimation)
        return selection

    def append_to_release(self, release, weight, feature_number, feature, effort_estimation):
        """
        Appends selected feature to mobile release plan.

        :param release: Release number
        :param weight: WAS
        :param feature_number: feature number
        :param feature: Feature number
        :param effort_estimation: Effort estimate
        """

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
        rows.append([features[i], release_1_object_score[i], release_2_object_score[i], release_3_object_score[i]])

    data = np.array(rows)
    result = pd.DataFrame(data=data)
    print(result)

    lp.assignment_function(rows)

    print(lp.mobile_release_plan)
    # print(lp.objective_function(was))
    print(lp.effort_release_1, lp.effort_release_2, lp.effort_release_3)


runner()
