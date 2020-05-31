from classification.boc_n_linguistic_model import SentenceClassification
from extraction.extract_app_features_safe import SAFE, ExtractionMode, nlp
import pandas as pd
import numpy as np


def get_value(thumbs_up_count):
    if thumbs_up_count > 30:
        return 5
    elif thumbs_up_count > 20:
        return 4
    elif thumbs_up_count > 15:
        return 3
    elif thumbs_up_count > 5:
        return 2
    else:
        return 1


def get_priority(score):
    if score == 1:
        return 5
    elif score == 2:
        return 4
    elif score == 3:
        return score
    elif score == 4:
        return 2
    else:
        return 1


def issue_tracker_features():
    df_dataset = pd.read_csv('../data/jira.catrob.at-PAINTROID-issues.csv')
    rows = []
    for index, data in df_dataset.iterrows():
        if data[68] == 'new' and data[43] == 'Story':
            # ['Feature Key',
            #     'Feature f(i)',
            #     'Effort(days) t(i,2)',
            #     'Stakeholder S (1), Value v(1,i)',
            #     'Stakeholder S (1), Urgency u(1,i)',
            #     'Stakeholder S (2), Value v(2,i)',
            #     'Stakeholder S (2), Urgency u(2,i)']
            feature = [data[149], data[30], int(data[28]), 0, '(0, 0, 0)', 0, '(0, 0, 0)']
            rows.append(feature)
    return rows


def get_features_from_app_reviews_and_issue_tracker():
    sc = SentenceClassification()
    cr = sc.classify_reviews()

    extraction_mode = ExtractionMode.USER_REVIEWS

    obj_safe = SAFE('PAINTROID', cr, extraction_mode, nlp)
    true_features_dict, extracted_features = obj_safe.get_reviews_with_extracted_features()
    dict_true_features = obj_safe.clean_features(true_features_dict)

    columns = ["Feature Key", "Feature f(i)", "Effort(days) t(i,2)", "Stakeholder S (1), Value v(1,i)",
               "Stakeholder S (1), Urgency u(1,i)", "Stakeholder S (2), Value v(2,i)",
               "Stakeholder S (2), Urgency u(2,i)"]
    rows = []
    for f_row in issue_tracker_features():
        rows.append(f_row)
    for dict in dict_true_features.items():
        if len(dict[1]['predicted_features']) != 0:
            for i, value in enumerate(dict[1]['predicted_features']):
                pf, pattern = value
                #     ['Feature Key',
                #     'Feature f(i)',
                #     'Effort(days) t(i,2)',
                #     'Stakeholder S (1), Value v(1,i)',
                #     'Stakeholder S (1), Urgency u(1,i)',
                #     'Stakeholder S (2), Value v(2,i)',
                #     'Stakeholder S (2), Urgency u(2,i)']
                feature = [
                    str(dict[0]) + '-' + str(i) + '-' + dict[1]['review-id'],
                    pf,
                    5,
                    get_value(int(dict[1]['thumbs_up_count'])),
                    '(' + str(get_priority(int(dict[1]['score']))) + ', 0, 0)',
                    0,
                    '(0, 0, 0)'
                ]
                rows.append(feature)

    df = pd.DataFrame(np.array(rows),
                      columns=columns)

    df.to_csv('../data/features.csv', index=False)
    print('here')


get_features_from_app_reviews_and_issue_tracker()
