import random

import pandas as pd


class MobileReleasePlanner(object):
    """Mobile Release Planning base."""

    def __init__(self, stakeholder_importance, release_relative_importance,
                 release_duration=14, coupling=None):
        """
        Initialize a genetic algorithm.

        :type stakeholder_importance:(int, int)
        :param stakeholder_importance (tuple): Stakeholders importance.

        :type release_relative_importance: (float, float, float)
        :param release_relative_importance: Release relative importance.

        :type release_duration : int
        :param release_duration: Release duration.

        :type coupling: {(str, str)}
        :param coupling: Coupled features.
        """

        if len(release_relative_importance) != 3:
            raise AttributeError("Pass a tuple of size 3 containing the each release importance!")
        if len(stakeholder_importance) != 2:
            raise AttributeError("Pass a tuple of size 2 containing the each stakeholders importance!")
        if coupling is None:
            coupling = {}
        if stakeholder_importance is None:
            self.stakeholder_importance = (6, 6)
        if release_relative_importance is None:
            self.release_relative_importance = (0.3, 0.3, 0.3)
        self.release_relative_importance = release_relative_importance
        self.stakeholder_importance = stakeholder_importance
        self.number_of_releases = 3
        self.effort_release_1 = 0.0
        self.effort_release_2 = 0.0
        self.effort_release_3 = 0.0
        self.coupling = coupling
        self.release_duration = release_duration
        self.inputs = pd.read_csv("../data/features-evaluation-clean.csv",
                                  dtype={"Stakeholder S (1), Value value(1,i)": "Int64",
                                         "Stakeholder S (2), Value value(2,i)": "Int64"})
        self.inputs.columns = ["Feature Key", "Feature f(i)", "Effort(Story Points) t(i,2)",
                               "Stakeholder S (1), Value v(1,i)",
                               "Stakeholder S (1), Urgency u(1,i)", "Stakeholder S (2), Value v(2,i)",
                               "Stakeholder S (2), Urgency u(2,i)"]
        features = self.inputs["Feature f(i)"].to_xarray().values
        self.keys = self.inputs["Feature Key"].to_xarray().values.tolist()
        self.effort = self.inputs["Effort(Story Points) t(i,2)"].to_xarray().values.tolist()
        self.results = []
        self.mobile_release_plan = []
        self.results.append(features.tolist())

    def calculate_was_for_all_features(self):
        """
        Calculates WAS for all features.

        """
        # self.print()
        for k in range(0, self.number_of_releases):
            row = []
            for index, data in self.inputs.iterrows():
                result = self.release_relative_importance[k] * (self.was(
                    [self.get_score(self.stakeholder_importance[0], data[3], self.urgency(vector=data[4], release=k)),
                     self.get_score(self.stakeholder_importance[1], data[5], self.urgency(vector=data[6], release=k))]))
                # (release, WAS, feature_key, feature, effort_estimation)
                row.append((k + 1, result, data[0], data[1], data[2]))
            self.results.append(row)

    def print(self):
        """
        Print shape and head of input csv file.

        """
        print(self.inputs.shape)
        print(self.inputs.head())

    @staticmethod
    def get_score(stakeholder_importance, value_on_feature, urgency_on_feature):
        """
        Stakeholders priorities.

        :param stakeholder_importance: A stakeholders importance
        :type stakeholder_importance: int
        :param value_on_feature: Stakeholders' value placed on feature(i)
        :type value_on_feature: int
        :param urgency_on_feature: Stakeholders' urgency placed on feature(i)
        :type urgency_on_feature: int
        :return: Product of Stakeholders' importance, value, and urgency.
        """
        return stakeholder_importance * value_on_feature * urgency_on_feature

    @staticmethod
    def urgency(vector, release):
        """
        Stakeholders' urgency on a release.

        :param vector: Urgency vector of feature(i)
        :type vector: str
        :param release: Release
        :type release: int
        :return: Urgency placed by stakeholder on release(r).
        """
        vector_tuple = tuple(
            map(lambda v: v, vector.replace("(", "").replace(")", "").replace(" ", "").replace("\'", "").split(",")))
        return int(vector_tuple[release])

    @staticmethod
    def was(scores):
        """
        Each value WAS(i, k) is determined as the weighted average of the products of
        the two dimensions of prioritization (stakeholder priorities).

        :param scores: Value of assigning feature(i) to release(r) for each stakeholder
        :type scores: list
        :return: Weighted Average Satisfaction(was) of assigning feature(i) to release(r).
        """
        was = 0.0
        for value in scores:
            was += value
        return was

    @staticmethod
    def objective_function(was, exclude_postponed=True):
        """
        An additive function exists in which the total objective function value is determined
        as the sum of the weighted average satisfaction WAS(i, k) of stakeholder priorities for
        all features f(i) when assigned to release k.

        Note: We consider a solution to be sufficiently good (or qualified) if it achieves
        at least 95 percent of the maximum objective function value.

        :param exclude_postponed: Exclude postponed was
        :type exclude_postponed: bool
        :param was: all WAS score of features for release plan (x)
        :type was: list
        :return: Objective function score.
        """
        fitness = 0.0
        for release, value, _, _, _ in was:
            if exclude_postponed and release == 4:
                continue
            fitness += value
        return round(fitness, 2)

    def can_assign_to_release(self, current_total_effort_of_release, effort_estimate, total_effort=None):
        """
        Assigns a feature to a mobile release plan or put in not feasible list if not feasible in current plan.

        :param current_total_effort_of_release: Current sum of effort in a release
        :type current_total_effort_of_release: float
        :param total_effort: Total effort estimate of coupled features
        :type total_effort: float
        :param effort_estimate: Estimate of a feature
        :type effort_estimate: float

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
        :type effort_estimate_1: float
        :param effort_estimate_2: Second effort estimate
        :type effort_estimate_2: float
        :return: Sum of two estimate values.
        """
        return effort_estimate_1 + effort_estimate_2

    def append_to_release(self, release, weight, feature_key, feature, effort_estimation, couple=None):
        """
        Appends selected feature to mobile release plan.

        :param couple: Coupled feature
        :type couple: tuple
        :param release: Release number
        :type release: int
        :param weight: WAS
        :type weight: float
        :param feature_key: Feature unique identifier
        :type feature_key: str
        :param feature: Feature key
        :type feature: str
        :param effort_estimation: Effort estimate
        :type effort_estimation: float
        """

        if couple is not None:
            self.mobile_release_plan.append((release, couple[1], couple[2], couple[3], couple[4]))
            self.mobile_release_plan.append((release, weight, feature_key, feature, effort_estimation))
            self.increase_effort(release, effort_estimation + couple[4])
        else:
            self.mobile_release_plan.append((release, weight, feature_key, feature, effort_estimation))
            self.increase_effort(release, effort_estimation)

    def increase_effort(self, release, effort_estimation):
        """
        Increase effort of a release.

        :param release: Release number
        :type release: int
        :param effort_estimation: Effort estimation
        :type effort_estimation: float
        """

        if release == 1:
            self.effort_release_1 += effort_estimation
        if release == 2:
            self.effort_release_2 += effort_estimation
        if release == 3:
            self.effort_release_3 += effort_estimation

    def is_coupled_with(self, feature_key):
        """
        Get other couple of a feature.

        :param feature_key: Feature unique identifier
        :type feature_key: str
        :return: Number of the feature's partner.
        """
        coupled_with = None
        for (f1, f2) in self.coupling:
            if feature_key == f1:
                coupled_with = f2
            elif feature_key == f2:
                coupled_with = f1
        return coupled_with

    def features(self):
        """
        All feature with calculate was

        :returns: Feature set with calculate was
        """
        self.calculate_was_for_all_features()

        rows = []
        # features = self.results[0]
        release_1_object_score = self.results[1]
        release_2_object_score = self.results[2]
        release_3_object_score = self.results[3]

        for i in range(0, len(self.keys)):
            rows.append([release_1_object_score[i], release_2_object_score[i], release_3_object_score[i]])
        return rows

    @staticmethod
    def get_random_was(feature_array, add_to_release=None):
        """
        Selects a WAS.

        :param add_to_release: Specify a release to add this feature
        :type add_to_release: int
        :param feature_array: WAS options
        :type feature_array: list
        :return: Any WAS.
        """
        index = random.randint(0, 2)
        if add_to_release is not None:
            _, score, key, description, effort = feature_array[index]
            return add_to_release, score, key, description, effort
        return feature_array[index]

    @staticmethod
    def get_max_was(feature_array, add_to_release=None):
        """
        Selects highest WAS.

        :param feature_array: WAS options
        :type feature_array: list
        :param add_to_release: Specify a release to add this feature
        :type add_to_release: int
        :return: Highest WAS.
        """
        sorted_features = sorted(feature_array, key=lambda s: s[1])
        selection = sorted_features[-1]
        if add_to_release is not None:
            _, score, key, description, effort = selection
            return add_to_release, score, key, description, effort
        return selection
