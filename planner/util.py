import pandas as pd
import numpy as np


def save_model_result(model, save_result_to):
    heading = ['Release', 'WAS', 'Key', 'Feature', 'Effort (Story Point)']

    mrp = model.mobile_release_plan

    rows = []

    for r in mrp:
        rows.append([r[0], r[1], r[2], r[3], r[4]])
    obj_score = model.objective_function(model.mobile_release_plan)

    rows.append(['', 'Effort R1: ' + str(model.effort_release_1),
                 'Effort R2: ' + str(model.effort_release_2),
                 'Effort R3: ' + str(model.effort_release_3),
                 'F(x): ' + str(obj_score)])

    df2 = pd.DataFrame(np.array(rows),
                       columns=heading)

    df2.to_csv(save_result_to)
    print('Done!!!')
