import time

import pandas as pd
import numpy as np
import random
import copy

from planner import base
from planner.util import save_model_result


class LP(base.MobileReleasePlanner):
    """Mobile Release Planning using Linear Programming."""

    def __init__(self, stakeholder_importance=None, release_relative_importance=None, release_duration=None,
                 coupling=None, highest=True, is_sorted=True):

        """
        Initialize a linear programming/greedy algorithm.

        :type is_sorted: bool
        :param is_sorted (bool): Sort features in descending order based on WAS of 1st release or shuffle.

        :type stakeholder_importance:(int, int)
        :param stakeholder_importance (tuple): Stakeholders importance.

        :type release_relative_importance: (float, float, float)
        :param release_relative_importance: Release relative importance.

        :type release_duration : int
        :param release_duration: Release duration.

        :type coupling: {(str, str)}
        :param coupling: Coupled features.

        :type highest: bool
        :param highest: Flag specified if feature should be chosen randomly or based on the highest WAS.
        """

        self.delete_flag = False
        self.highest = highest
        self.is_sorted = is_sorted
        self.start = None
        self.end = None

        super(LP, self).__init__(stakeholder_importance, release_relative_importance, release_duration, coupling)

    def assignment_function(self, array_was_feature):
        """
        Greedy function for feature assignment.

        :type array_was_feature: list
        :param array_was_feature: Release and WAS for a feature
        """
        self.start = time.time()
        # original_feature_set = copy.copy(array_was_feature)
        if self.is_sorted:
            array_was_feature = sorted(array_was_feature, key=lambda f: f[0][1], reverse=True)
        else:
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

        self.end = time.time()

    def assign(self, max_feature, feature_array, total_effort=None, couple=None, couple_array=None):
        """
        Assigns a feature to a mobile release plan or put in not feasible list if not feasible in current plan.

        :param max_feature: Feature with highest WAS
        :type max_feature: list
        :param feature_array: Feature details
        :type feature_array: list
        :param total_effort: Total effort estimate of coupled features
        :type total_effort: float
        :param couple: Feature partner (couple)
        :type couple: tuple
        :param couple_array: Couple details
        :type couple_array: list
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


def test():
    # coupling = {("F7", "F8"), ("F9", "F12"), ("F13", "F14")}

    d = 27
    si = [(4, 6), (6, 4)]
    rrp = [(0.3, 0.3, 0.3), (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)]

    for i in si:
        for ri in rrp:

            lp = LP(coupling=None, stakeholder_importance=i, release_relative_importance=ri,
                    release_duration=d, is_sorted=False)

            features = lp.features()

            lp.assignment_function(features)

            save_model_result(lp, 'lp-test-results/' + 'duration-' + str(d) + '-stakeholder_importance-' + ''.join(
                str(i)) + '-release_relative_importance-' + ''.join(str(ri)) + '.csv')


def run():

    d = 27
    i = (6, 4)
    ri = (0.8, 0.1, 0.1)

    lp = LP(coupling=None, stakeholder_importance=i, release_relative_importance=ri,
            release_duration=d, is_sorted=False)

    features = lp.features()

    lp.assignment_function(features)

    processing_time = lp.end - lp.start
    print('Processing Time: ' + str(processing_time))

    save_model_result(lp, 'results/mrp_lp.csv')


def main():

    run()


if __name__ == "__main__":
    main()
