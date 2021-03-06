from textwrap import wrap

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from similarity.setup import cos_sim, get_jaccard_sim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)

plt.style.use('seaborn-whitegrid')

model_names = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens',
               'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'jaccard']
features = pd.read_csv("labelled-feature.csv")
features.columns = ["Feature 1", "Feature 2", "Label"]

thresholds = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
              0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
              0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
              0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
              0.90, 0.91, 0.92, 0.93, 0.94, 0.95]

heading = ['Feature 1', 'Feature 2', 'Label', 'Cosine Similarity', 'Predicted Label']
heading2 = ['Threshold', 'Accuracy', 'Precision', 'Recall', 'f1 Score']


def exp(threshold, model_name, model):
    rows = []
    true_label = []
    predicted_label = []
    temp = {}
    if model_name != 'jaccard':
        for i, f in features.iterrows():
            if f['Feature 1'] in temp:
                emb1 = temp[f['Feature 1']]
            else:
                emb1 = model.encode([f['Feature 1']])[0]
                temp[f['Feature 1']] = emb1
            if f['Feature 2'] in temp:
                emb2 = temp[f['Feature 2']]
            else:
                emb2 = model.encode([f['Feature 2']])[0]
                temp[f['Feature 2']] = emb2

            similarity_list = cos_sim(emb1, emb2)
            similarity = similarity_list[0][1]
            pred = 'similar' if similarity >= threshold else 'different'

            true_label.append(0 if f['Label'] == 'similar' else 1)
            predicted_label.append(0 if pred == 'similar' else 1)

            rows.append([f['Feature 1'], f['Feature 2'], f['Label'], similarity, pred])

        accuracy = accuracy_score(true_label, predicted_label)
        precision = precision_score(true_label, predicted_label)
        recall = recall_score(true_label, predicted_label)
        f1score = f1_score(true_label, predicted_label)
    else:
        for i, f in features.iterrows():
            similarity = get_jaccard_sim(f['Feature 1'], f['Feature 2'])
            pred = 'similar' if similarity >= threshold else 'different'
            rows.append([f['Feature 1'], f['Feature 2'], f['Label'], similarity, pred])

            true_label.append(0 if f['Label'] == 'similar' else 1)
            predicted_label.append(0 if pred == 'similar' else 1)

        accuracy = accuracy_score(true_label, predicted_label)
        precision = precision_score(true_label, predicted_label)
        recall = recall_score(true_label, predicted_label)
        f1score = f1_score(true_label, predicted_label)

        heading[3] = 'Jaccard Similarity'
    df2 = pd.DataFrame(np.array(rows),
                       columns=heading)

    df2.to_csv('results/results-model-' + model_name + '-threshold-' + str(threshold) + '.csv')
    return accuracy, precision, recall, f1score


def plot(precision, f1score, recall, model):
    df = pd.DataFrame({'x': thresholds, 'y1': precision, 'y2': f1score, 'y3': recall})

    plt.title("\n".join(wrap("Threshold Plot for " + model, 60)))
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')

    plt.plot('x', 'y1', data=df, color='blue', linewidth=2, label="Precision")
    plt.plot('x', 'y2', data=df, color='brown', linewidth=2, label="F1 Score")
    plt.plot('x', 'y3', data=df, color='black', linewidth=2, label="Recall")
    plt.legend(prop={'size': 10})
    plt.savefig('results/' + model + '.png')
    plt.show()


def get_plots_for_models():
    columns = ["", "Threshold", "Accuracy", "Precision", "Recall", "f1 Score"]
    scores_model_bert_base_nli_mean_tokens = pd.read_csv("results/scores-model-bert-base-nli-mean-tokens.csv")
    scores_model_bert_base_nli_stsb_mean_tokens = pd.read_csv("results/scores-model-bert-base-nli-stsb-mean-tokens.csv")
    scores_model_bert_large_nli_mean_tokens = pd.read_csv("results/scores-model-bert-large-nli-mean-tokens.csv")
    scores_model_bert_large_nli_stsb_meantokens = pd.read_csv("results/scores-model-bert-large-nli-stsb-mean-tokens.csv")
    scores_model_jaccard = pd.read_csv("results/scores-model-jaccard.csv")

    scores_model_bert_base_nli_mean_tokens.columns = columns
    scores_model_bert_base_nli_stsb_mean_tokens.columns = columns
    scores_model_bert_large_nli_mean_tokens.columns = columns
    scores_model_bert_large_nli_stsb_meantokens.columns = columns
    scores_model_jaccard.columns = columns

    loop_data('M1 (bert-base-nli-mean-tokens + Cosine Similarity)', scores_model_bert_base_nli_mean_tokens)
    loop_data('M3 (bert-base-nli-stsb-mean-tokens + Cosine Similarity)', scores_model_bert_base_nli_stsb_mean_tokens)
    loop_data('M2 (bert-large-nli-mean-tokens + Cosine Similarity)', scores_model_bert_large_nli_mean_tokens)
    loop_data('M4 (bert-large-nli-stsb-mean-tokens + Cosine Similarity).', scores_model_bert_large_nli_stsb_meantokens)
    loop_data('M5 (Jaccard Similarity)', scores_model_jaccard)


def loop_data(model, data):
    precision = []
    f1score = []
    recall = []
    for i, r in data.iterrows():
        precision.append(r['Precision'])
        f1score.append(r['f1 Score'])
        recall.append(r['Recall'])

    plot(precision, f1score, recall, model)


def run():

    for n in model_names:
        rows = []
        model = None
        if n != 'jaccard':
            model = SentenceTransformer(n)
        for t in thresholds:
            accuracy, precision, recall, f1score = exp(t, n, model)
            rows.append([t, accuracy, precision, recall, f1score])
        df2 = pd.DataFrame(np.array(rows),
                           columns=heading2)

        df2.to_csv('results/scores-model-' + n + '.csv')


# run()
get_plots_for_models()



