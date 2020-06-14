import gensim
import nltk
import logging
import math

from classification.boc_n_linguistic_model import SentenceClassification
from extraction.extract_app_features_safe import SAFE, ExtractionMode, nlp
from sentence_transformers import SentenceTransformer
from similarity.setup import cos_sim
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
nltk.download('wordnet')

THRESHOLD = 0.710855
model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')


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


def consolidate(rows_jira, rows_app_store, calculate_from_review_values=True):
    consolidated = []
    merge = []
    for row_j in rows_jira:
        for i, row_r in enumerate(rows_app_store):
            merge.append((row_j, (i, row_r)))
    similar = {}
    temp = {}
    for m_row1, m_row2 in merge:
        if m_row1[1] in temp:
            emb1 = temp[m_row1[1]]
        else:
            emb1 = model.encode([m_row1[1]])[0]
            temp[m_row1[1]] = emb1
        if m_row2[1][1] in temp:
            emb2 = temp[m_row2[1][1]]
        else:
            emb2 = model.encode([m_row2[1][1]])[0]
            temp[m_row2[1][1]] = emb2

        similarity_list = cos_sim(emb1, emb2)
        similarity = similarity_list[0][1]

        if similarity > THRESHOLD:
            if m_row1[1] in similar:
                similar[m_row1[1]].append(m_row2)
            else:
                similar[m_row1[1]] = [m_row1, m_row2]

    for key, item in similar.items():
        count = len(item) - 1
        value = 0
        priority = 0
        for i, f in enumerate(item):
            if i == 0:
                continue
            value += f[1][3]
            vector_tuple = get_priority_tuple(f[1][4])
            priority += int(vector_tuple[0])
            del rows_app_store[f[0]]
        s_1_value = math.ceil(value / count)
        s_1_priority = math.ceil(priority / count)
        item[0][3] = s_1_value
        item[0][4] = '(' + str(s_1_priority) + ', 0, 0)'
        if calculate_from_review_values:
            item[0][5] = 1 if math.ceil(0.5 * s_1_value) == 0 else math.ceil(0.5 * s_1_value)
            item[0][6] = '(' + str(
                1 if math.ceil(0.5 * s_1_priority) == 0 else math.ceil(0.5 * s_1_priority)) + ', 0, 0)'
        consolidated.append(item[0])

    for f in rows_app_store:
        if calculate_from_review_values:
            f[5] = 1 if math.ceil(0.5 * int(f[3])) == 0 else math.ceil(0.5 * int(f[3]))
            f[6] = '(' + str(1 if math.ceil(0.5 * int(get_priority_tuple(f[4])[0])) == 0 else math.ceil(
                0.5 * int(get_priority_tuple(f[4])[0]))) + ', 0, 0)'
        consolidated.append(f)

    for f in rows_jira:
        if f[1] in similar:
            continue
        if calculate_from_review_values:
            f[3] = 1
            f[4] = '(1, 0, 0)'
            f[5] = 1
            f[6] = '(1, 0, 0)'
        consolidated.append(f)

    return consolidated


def get_priority_tuple(priority_vec):
    return tuple(
        map(lambda v: v,
            priority_vec.replace("(", "").replace(")", "").replace(" ", "").replace("\'", "").split(",")))


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
                    avg_s_1_v = math.ceil((existing_f[0][1][3] + value) / 2)
                    vector_tuple = get_priority_tuple(existing_f[0][1][4])
                    avg_s_1_p = math.ceil((int(vector_tuple[0]) + priority) / 2)
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

    result = consolidate(rows_jira, rows_app_store)

    df = pd.DataFrame(np.array(result),
                      columns=columns)

    df.to_csv('../data/features.csv', index=False)
    print('here')


get_features_from_app_reviews_and_issue_tracker()
