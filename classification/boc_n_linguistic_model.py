import ast
import csv
import time
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


class Source(Enum):
    JIRA = 'JIRA'
    GITHUB = 'GITHUB'
    PLAY_STORE = 'PLAY_STORE'


class SentenceClassification:
    def __init__(self):
        df_dataset = pd.read_csv('../Review_Dataset/Gu_Dataset_features.csv', index_col=False)
        self.review_dataset = pd.read_csv('../data/reviews.csv', index_col=False,
                                          dtype={"class": "string"})
        self.train_dataset, self.test_data = train_test_split(df_dataset, test_size=0.2)

    def evaluate_model_on_test_set(self, ds_test, classify_model, vec, is_actual_test_review=False):

        test_feature_dicts = []
        true_labels = []

        for index, row in ds_test.iterrows():
            features = self.sentence_features(row)
            test_feature_dicts.append(features)
            true_labels.append(row['class'])

        test_x = vec.transform(test_feature_dicts)

        # print(true_labels)

        # Predict the test data labels
        pred_y = classify_model.predict(test_x)

        if is_actual_test_review:
            return pred_y
        else:
            _precision, _recall, f_score, support = precision_recall_fscore_support(true_labels, pred_y,
                                                                                    labels=['E', 'P', 'R', 'B', 'N'])
            return _precision, _recall, f_score

    def model(self):

        train_x, feature_transform, class_labels = self.generate_feature_matrix(self.train_dataset)

        train_labels = np.array(class_labels)
        lr_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=.09)

        lr_model.fit(train_x, train_labels)

        return feature_transform, lr_model

    def train_n_evaluate_on_test_data(self):

        feature_transform, lr_model = self.model()
        _precision, _recall, f_score = self.evaluate_model_on_test_set(self.test_data, lr_model, feature_transform)

        return _precision, _recall, f_score

    @staticmethod
    def sentence_features(row):

        sentence = ast.literal_eval((row['tagged_sent']))

        # pos tag features
        pos_tags_feature = '-'.join(tag for (word, tag) in sentence)

        all_char_grams = ast.literal_eval(row['char_grams'])

        features = {}

        for char_gram in all_char_grams:
            features['{}'.format(char_gram)] = True

        # root word of a sentence

        features['ROOT({})'.format(row['root'])] = True

        features['{}'.format(pos_tags_feature)] = True

        features['{}'.format(row['parse_tree'])] = True

        features['{}'.format(row['dep_tree'])] = True

        return features

    def generate_feature_matrix(self, ds):
        feature_dicts = []
        class_labels = []

        for index, row in ds.iterrows():
            features = self.sentence_features(row)
            feature_dicts.append(features)
            class_labels.append(row['class'])

        assert len(feature_dicts) == ds.shape[0]

        vec = DictVectorizer()
        feature_matrix = vec.fit_transform(feature_dicts)
        return feature_matrix, vec, class_labels

    def classify_reviews(self):
        classified_reviews = {}
        feat_transform, model = self.model()
        reviews = self.review_dataset
        predictions = self.evaluate_model_on_test_set(reviews, model, feat_transform,
                                                      is_actual_test_review=True)

        predictions = predictions.tolist()

        for index, row in reviews.iterrows():
            review_class = predictions[index]
            if review_class == 'N' or review_class == 'P' or review_class == 'B':
                continue

            review_dict = {
                'review-id': row['reviewId'],
                'class': review_class,
                'username': row['userName'],
                'review_sent': row['sentence'],
                'score': row['score'],
                'thumbs_up_count': row['thumbsUpCount'],
                'review_created_version': row['reviewCreatedVersion'],
                'comment_made_on': row['at'],
                'reply_content': row['replyContent'],
                'replied_at': row['repliedAt'],
                'true_features': [],
                'predicted_features': [],
                'source': Source.PLAY_STORE
            }
            classified_reviews[index] = review_dict
        return classified_reviews


def evaluate():
    _start_time = time.time()
    sc = SentenceClassification()
    max_iteration = 10

    average_precision_eval, average_recall_eval, average_fscore_eval = ([] for i in range(3))
    average_precision_praise, average_recall_praise, average_fscore_praise = ([] for i in range(3))
    average_precision_request, average_recall_request, average_fscore_request = ([] for i in range(3))
    average_precision_bug, average_recall_bug, average_fscore_bug = ([] for i in range(3))
    average_precision_others, average_recall_others, average_fscore_others = ([] for i in range(3))

    with open('BoC_n_Linguistic_Model_Results.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Iteration', 'P', 'R', 'F1', 'P', 'R', 'F1', 'P', 'R', 'F1', 'P', 'R', 'F1', 'P', 'R', 'F1'])

        for i in range(0, max_iteration):
            print('############Iteration # %d###############' % (i + 1))

            # obj.SplitDataintoTrainTest()
            precision, recall, fscore = sc.train_n_evaluate_on_test_data()

            average_precision_eval.append(precision[0])
            average_recall_eval.append(recall[0])
            average_fscore_eval.append(fscore[0])

            average_precision_praise.append(precision[1])
            average_recall_praise.append(recall[1])
            average_fscore_praise.append(fscore[1])

            average_precision_request.append(precision[2])
            average_recall_request.append(recall[2])
            average_fscore_request.append(fscore[2])

            average_precision_bug.append(precision[3])
            average_recall_bug.append(recall[3])
            average_fscore_bug.append(fscore[3])

            average_precision_others.append(precision[4])
            average_recall_others.append(recall[4])
            average_fscore_others.append(fscore[4])

            lst = [i + 1, precision[0], recall[0], fscore[0], precision[1], recall[1], fscore[1]]
            lst += [precision[2], recall[2], fscore[2], precision[3], recall[3], fscore[3]]
            lst += [precision[4], recall[4], fscore[4]]

            writer.writerow(lst)

            print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (feature evaluation)" % (
                precision[0], recall[0], fscore[0]))
            print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (praise)" % (precision[1], recall[1], fscore[1]))
            print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (feature request)" % (
                precision[2], recall[2], fscore[2]))
            print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (bug report)" % (precision[3], recall[3], fscore[3]))
            print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (others)" % (precision[4], recall[4], fscore[4]))

            print('+++++++++++++++ITERATION FINISHED++++++++++++++++++++++++++++')

        print("###########Average results for each class over 10 iterations####################")

        print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (feature evaluation)" % (
            np.mean(average_precision_eval), np.mean(average_recall_eval), np.mean(average_fscore_eval)))
        print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (praise)" % (
            np.mean(average_precision_praise), np.mean(average_recall_praise), np.mean(average_fscore_praise)))
        print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (feature request)" % (
            np.mean(average_precision_request), np.mean(average_recall_request), np.mean(average_fscore_request)))
        print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (bug report)" % (
            np.mean(average_precision_bug), np.mean(average_recall_bug), np.mean(average_fscore_bug)))
        print("Precision : %.3f, Recall : %.3f , Fscore : %.3f (others)" % (
            np.mean(average_precision_others), np.mean(average_recall_others), np.mean(average_fscore_others)))

        print('The script finished in {0} seconds'.format(time.time() - _start_time))


if __name__ == '__main__':
    # evaluate()
    start_time = time.time()
    sentence_classifier = SentenceClassification()
    cr = sentence_classifier.classify_reviews()

    end_time = time.time()

    print('done')
