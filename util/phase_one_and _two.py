import gensim
import nltk

from classification.boc_n_linguistic_model import SentenceClassification
from extraction.extract_app_features_safe import SAFE, ExtractionMode, nlp
import pandas as pd
import numpy as np

nltk.download('wordnet')


def get_value(score):
    return score


def get_priority(thumbs_up_count):
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


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return ' '.join(result)


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
            feature = [data[149], preprocess(data[30]), int(data[28]), 0, '(0, 0, 0)', 0, '(0, 0, 0)']
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
    rows_jira = []
    rows_app_store = []
    for f_row in issue_tracker_features():
        rows_jira.append(f_row)

    for dict in dict_true_features.items():
        if len(dict[1]['predicted_features']) != 0:
            for i, value in enumerate(dict[1]['predicted_features']):
                pf, pattern = value
                existing_f = [(idx, feature) for idx, feature in enumerate(rows_app_store) if
                              (feature[1] == pf)]
                value = get_value(int(dict[1]['score']))
                priority = get_priority(int(dict[1]['thumbs_up_count']))
                if len(existing_f) == 1:
                    avg_s_1_v = round((existing_f[0][1][3] + value) / 2)
                    vector_tuple = tuple(
                        map(lambda v: v,
                            existing_f[0][1][4].replace("(", "").replace(")", "").replace(" ", "").replace("\'", "").split(",")))
                    avg_s_1_p = round((int(vector_tuple[0]) + priority) / 2)
                    value = avg_s_1_v
                    priority = avg_s_1_p
                    del rows_app_store[existing_f[0][0]]
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
                    value,
                    '(' + str(priority) + ', 0, 0)',
                    0,
                    '(0, 0, 0)'
                ]
                rows_app_store.append(feature)

    df = pd.DataFrame(np.array(rows_jira + rows_app_store),
                      columns=columns)

    df.to_csv('../data/features.csv', index=False)
    print('here')


get_features_from_app_reviews_and_issue_tracker()
