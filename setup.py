import pandas as pd
import numpy as np


class LPReleasePlanner(object):
    stakeholder_importance = None
    release_relative_importance = None
    number_of_releases = None
    coupling = None
    precedence = None
    results = []

    def __init__(self, stakeholder_importance=(4, 6), release_relative_importance=(0.7, 0.3, 0.0), number_of_releases=3,
                 coupling=None,
                 precedence=None):
        if precedence is None:
            precedence = {}
        if coupling is None:
            coupling = {}
        self.stakeholder_importance = stakeholder_importance
        self.release_relative_importance = release_relative_importance
        self.number_of_releases = number_of_releases
        self.coupling = coupling
        self.precedence = precedence
        self.inputs = pd.read_csv("data/sample.csv", skiprows=1, nrows=15,
                                  dtype={"Value value(1,i)": "Int64", "Value value(2,i)": "Int64"})
        self.inputs.columns = ["Feature f(i)", "Analyst & designers (hrs) r(i,1)", "Developers (hrs) r(i,2)",
                               "QA (hrs) r(i,3)", "Budget (US$ in thousands)", "Value value(1,i)",
                               "Urgency urgency(1,i)", "Value value(2,i)", "Urgency urgency(2,i)"]
        features = self.inputs["Feature f(i)"].to_xarray().values
        self.results.append(features.tolist())

    def objective_function(self):
        self.print()
        for k in range(0, self.number_of_releases):
            row = []
            for index, data in self.inputs.iterrows():
                result = self.release_relative_importance[k] * (self.get_score(
                    [self.was(self.stakeholder_importance[0], data[5], self.urgency(vector=data[6], release=k)),
                     self.was(self.stakeholder_importance[1], data[7], self.urgency(vector=data[8], release=k))]))
                row.append((k + 1, result))
            self.results.append(row)

    @staticmethod
    def get_score(was):
        score = 0.0
        for value in was:
            score += value
        return score

    # weighted average satisfaction WAS(i, k) of stakeholder priorities for
    # all features f(i) when assigned to release k.
    @staticmethod
    def was(stakeholder_importance, value_on_feature, urgency_on_feature):
        return stakeholder_importance * value_on_feature * urgency_on_feature

    @staticmethod
    def urgency(vector, release):
        vector_tuple = tuple(
            map(lambda v: v, vector.replace("(", "").replace(")", "").replace(" ", "").replace("\'", "").split(",")))
        return int(vector_tuple[release])

    def print(self):
        print(self.inputs.shape)
        print(self.inputs.head())


def runner():
    coupling = {(7, 8), (9, 12), (13, 14)}
    precedence = {(2, 1), (5, 6), (3, 11), (8, 9), (13, 15)}

    lp = LPReleasePlanner(coupling=coupling, precedence=precedence)
    lp.objective_function()

    rows = []
    features = lp.results[0]
    release_1_object_score = lp.results[1]
    release_2_object_score = lp.results[2]

    for i in range(0, 15):
        rows.append([features[i], release_1_object_score[i], release_2_object_score[i]])

    data = np.array(rows)
    result = pd.DataFrame(data=data)
    print(result)


runner()
