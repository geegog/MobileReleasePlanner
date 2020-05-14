import pandas as pd
import numpy as np
import random
import copy

from planner import base


class LP(base.MobileReleasePlanner):

    def __init__(self, stakeholder_importance=None, release_relative_importance=None, release_duration=None,
                 coupling=None, highest=True):

        self.delete_flag = False
        self.highest = highest

        super(LP, self).__init__(stakeholder_importance, release_relative_importance, release_duration, coupling)

    def assignment_function(self, array_was_feature):
        """
        Greedy function for feature assignment.

        :param array_was_feature: Release and WAS for a feature
        """

        original_feature_set = copy.copy(array_was_feature)
        random.shuffle(array_was_feature)
        for feature_array in array_was_feature:
            if feature_array is not None:
                if self.highest:
                    max_feature = self.get_max_was(feature_array)
                else:
                    max_feature = self.get_random_was(feature_array)
                couple_key = self.is_coupled_with(max_feature[2])
                if couple_key is not None:
                    feature = [(idx, feature) for idx, feature in enumerate(array_was_feature) if
                               (feature is not None and feature[0][2] == couple_key)]
                    if self.highest:
                        partner = self.get_max_was(feature[0][1])
                    else:
                        partner = self.get_random_was(feature[0][1])
                    total_effort = self.sum_couple_effort(max_feature[4], partner[4])
                    self.assign(max_feature, feature_array, total_effort, partner,
                                array_was_feature[feature[0][0]])
                    if self.delete_flag:
                        index = [idx for idx, f in enumerate(array_was_feature) if
                                 (f is not None and f[0][2] == couple_key)]
                        array_was_feature[index[0]] = None
                        self.delete_flag = False
                else:
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
            if couple is not None:
                self.append_to_release(4, max_feature[1], max_feature[2], max_feature[3], max_feature[4], couple)
                self.delete_flag = True
            else:
                self.append_to_release(4, max_feature[1], max_feature[2], max_feature[3], max_feature[4])


def runner():
    coupling = {("F7", "F8"), ("F9", "F12"), ("F13", "F14")}

    lp = LP(coupling=coupling, stakeholder_importance=(4, 6), release_relative_importance=(0.3, 0.0, 0.7),
            release_duration=14)

    features = lp.features()

    # data = np.array(features)
    # result = pd.DataFrame(data=data)
    # print(result)

    lp.assignment_function(features)

    print(lp.mobile_release_plan)
    print(lp.objective_function(lp.mobile_release_plan))
    print(lp.effort_release_1, lp.effort_release_2, lp.effort_release_3)


runner()
