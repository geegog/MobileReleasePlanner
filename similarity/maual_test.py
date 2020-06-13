import pandas as pd
import numpy as np
import logging

from sentence_transformers import SentenceTransformer
from similarity.setup import cos_sim, get_jaccard_sim
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)

bbnmt = 'bert-base-nli-mean-tokens'
blnmt = 'bert-large-nli-mean-tokens'
bbnsmt = 'bert-base-nli-stsb-mean-tokens'
blnsmt = 'bert-large-nli-stsb-mean-tokens'

model = SentenceTransformer(blnsmt)

features = pd.read_csv("labelled-feature.csv")
features.columns = ["Feature 1", "Feature 2", "Label"]
threshold = 0.710854647

heading = ['Feature 1', 'Feature 2', 'Label', 'Cosine Similarity', 'Predicted Label']
rows = []
true_label = []
predicted_label = []
temp = {}
for i, f in features.iterrows():
    emb1 = None
    if f['Feature 1'] in temp:
        emb1 = temp[f['Feature 1']]
    else:
        emb1 = model.encode([f['Feature 1']])[0]
        temp[f['Feature 1']] = emb1
    emb2 = None
    if f['Feature 2'] in temp:
        emb2 = temp[f['Feature 2']]
    else:
        emb2 = model.encode([f['Feature 2']])[0]
        temp[f['Feature 2']] = emb2

    similarity_list = cos_sim(emb1, emb2)
    similarity = similarity_list[0][1]
    pred = 'similar' if similarity > threshold else 'different'

    true_label.append(1 if f['Label'] == 'similar' else 0)
    predicted_label.append(1 if pred == 'similar' else 0)

    rows.append([f['Feature 1'], f['Feature 2'], f['Label'], similarity, pred])

# j = 'jaccard'
# for i, f in features.iterrows():
#     similarity = get_jaccard_sim(f['Feature 1'], f['Feature 2'])
#     pred = 'similar' if similarity > threshold else 'different'
#     rows.append([f['Feature 1'], f['Feature 2'], f['Label'], similarity, pred])
#
#     true_label.append(1 if f['Label'] == 'similar' else 0)
#     predicted_label.append(1 if pred == 'similar' else 0)

print(blnsmt, " accuracy: ", accuracy_score(true_label, predicted_label))

df2 = pd.DataFrame(np.array(rows),
                   columns=heading)

df2.to_csv('results-' + blnsmt + '.csv')
