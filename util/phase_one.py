
from classification.boc_n_linguistic_model import SentenceClassification
from extraction.extract_app_features_safe import SAFE, ExtractionMode, nlp


def get_predicted_features():
    sc = SentenceClassification()
    cr = sc.classify_reviews()

    extraction_mode = ExtractionMode.USER_REVIEWS

    obj_safe = SAFE('PAINTROID', cr, extraction_mode, nlp)
    true_features_dict, extracted_features = obj_safe.get_reviews_with_extracted_features()
    dict_true_features = obj_safe.clean_features(true_features_dict)
    print('here')


get_predicted_features()
