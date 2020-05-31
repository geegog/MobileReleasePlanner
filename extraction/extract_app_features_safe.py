import importlib
import re
from enum import Enum

import nltk
import spacy
from nltk.stem.snowball import SnowballStemmer

import extraction.safepatterns
import extraction.text_pre_processing
from extraction.text_pre_processing import TextProcessing

nltk.download('universal_tagset')

importlib.reload(extraction.safepatterns)
importlib.reload(extraction.text_pre_processing)

stemmer = SnowballStemmer("english")
nlp = spacy.load('en_core_web_sm')


class ExtractionMode(Enum):
    APP_DESCRIPTION = 1
    USER_REVIEWS = 2


class EvaluationType(Enum):
    EXACT_TOKEN = 1
    PARTIAL_TOKEN = 2
    EXACT_TYPE = 3
    PARTIAL_TYPE = 4
    SUBSET_TOKEN = 5


class SAFE:
    def __init__(self, app_name, review_sents, extract_mode, nlp):
        self.reviews_with_sents_n_features = {}
        self.nlp = nlp
        self.data = review_sents
        self.appName = app_name
        self.extract_mode = extract_mode

    def get_reviews_with_extracted_features(self):
        self.preprocess_data()
        return self.data, self.extracted_app_features_reviews

    def preprocess_data(self):

        if self.extract_mode == ExtractionMode.APP_DESCRIPTION:
            sents_segmented = True
        else:
            sents_segmented = False

        count = 0

        for review_id in self.data.keys():
            review_sent_text = self.data[review_id]['review_sent']

            sents_with_features = {}
            review_sent_wise_features = []

            text_processor = TextProcessing(self.appName, review_sent_text)
            unclean_sents = text_processor.segment_into_sentences(sents_segmented)
            review_clean_sentences = text_processor.get_clean_sentences()
            safe_patterns_obj = extraction.safepatterns.SafePatterns(self.appName, review_id, review_clean_sentences,
                                                                     unclean_sents)
            sents_with_features = safe_patterns_obj.extract_features_analyzing_sent_pos_patterns()

            for sid in sents_with_features.keys():
                review_sent_wise_features.extend(sents_with_features[sid]['extracted_features'])

            self.reviews_with_sents_n_features[review_id] = sents_with_features
            self.data[review_id]['predicted_features'] = review_sent_wise_features

            count = count + 1

        self.extracted_app_features_reviews = self.get_list_of_extracted_app_features()

    def clean_features(self, true_features_dict):

        app_words = self.appName.lower().split('_')

        # remove noise
        for review_id in true_features_dict.keys():
            # found_index=-1
            review_level_features_info = true_features_dict[review_id]
            review_text = review_level_features_info['review_sent']
            lst_extracted_feature = true_features_dict[review_id]['predicted_features']
            list_clean_feaures = []
            for findex, feature_info in enumerate(lst_extracted_feature):
                regex = re.compile('[@_!#$%^&*()-<>?/\|}{~:]')
                contain_special_character = False
                if regex.search(feature_info[0]) is None:
                    contain_special_character = False
                else:
                    contain_special_character = True

                words = feature_info[0].split()
                parse_feature = nlp(feature_info[0])
                duplicate_words = all(stemmer.stem(x) == stemmer.stem(words[0]) for x in words)
                contain_pronoun = any([w.tag_ == 'PRP' for w in parse_feature])
                contain_punct = any([w.tag_ == 'LS' for w in parse_feature])
                lemma_app_words = [w.lemma_ for w in parse_feature]
                contain_app_words = all(w in lemma_app_words for w in app_words)

                if contain_pronoun is not True and duplicate_words != True and contain_punct != True and contain_app_words != True and contain_special_character != True:
                    list_clean_feaures.append(feature_info)

            clean_features_list = list_clean_feaures.copy()

            # if shorter extracted app features are subsequence of a longer aspect term , remove shorter

            for feature_term1 in list_clean_feaures:
                status = any([feature_term1[0] in f[0] for f in list_clean_feaures if f[0] != feature_term1[0]])
                if status:
                    clean_features_list.remove(feature_term1)

            true_features_dict[review_id]['predicted_features'] = list(set(clean_features_list))

        return true_features_dict

    def get_list_of_extracted_app_features(self):
        list_extracted_app_features = []
        for sent_id in self.reviews_with_sents_n_features.keys():
            sents_with_app_features = self.reviews_with_sents_n_features[sent_id]
            for sent_id in sents_with_app_features.keys():
                app_features = sents_with_app_features[sent_id]['extracted_features']
                list_extracted_app_features.extend(app_features)

        return list_extracted_app_features
