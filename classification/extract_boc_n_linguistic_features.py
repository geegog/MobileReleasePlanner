from os import listdir
from os.path import isfile, join

import nltk
import pandas as pd
import requests
import spacy
from nltk.tree import Tree

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')
url = 'http://localhost:9000/?'


class ReviewFeatureExtractor:

    def __init__(self):
        self.df_dataset = pd.DataFrame(columns=('class', 'sentence'))

    def extract_boc_n_linguistic_features(self):
        dataset_path = "../Review_Dataset"

        file_list = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

        for file_path in file_list:
            print(file_path)
            file_full_path = dataset_path + "/" + file_path
            df = pd.read_csv(file_full_path)
            class_type = []
            self.get_dataset(df=df, content_index=1, is_eval=True)

        self.df_dataset.to_csv('../Review_Dataset/Gu_Dataset_features.csv', index=True)

    def extract_boc_n_linguistic_features_from_app_reviews(self):

        file_full_path = "../data/org.catrobat.paintroid.csv"
        df = pd.read_csv(file_full_path)

        self.get_dataset(df=df, content_index=4)

        self.df_dataset.to_csv('../data/reviews.csv', index=True)

    @staticmethod
    def clean_sentence(sent):
        typos = [["im", "i m", "I m"], ['Ill'], ["cant", "cnt"], ["doesnt"], ["dont"], ["isnt"], ["didnt"], ["couldnt"],
                 ["Cud"], ["wasnt"], ["wont"], ["wouldnt"], ["ive"], ["Ive"], ["theres"], ["awsome", "awsum", "awsm"],
                 ["Its"], ["dis", "diz"], ["plzz", "Plz ", "plz", "pls ", "Pls", "Pls "], [" U ", " u "], ["ur"], ["b"],
                 ["r"], ["nd ", "n", "&"], ["bt"], ["nt"], ["coz", "cuz"], ["jus", "jst"], ["luv", "Luv"], ["gud"],
                 ["Gud"], ["wel"], ["gr8", "Gr8", "Grt", "grt"], ["Gr\\."], ["pics"], ["pic"], ["hav"], ["nic"],
                 ["nyc ", "9ce"], ["Nic"], ["agn"], ["thx", "tks", "thanx"], ["Thx"], ["thkq"], ["Xcellent"], ["vry"],
                 ["Vry"], ["fav", "favo"], ["soo", "sooo", "soooo"], ["ap"], ["b4"], ["ez"], ["w8"], ["msg"], ["alot"],
                 ["lota"], ["kinda"], ["omg"], ["gota"]]
        replacements = ["I'm", "i will", "can't", "doesn't", "don't", "isn't", "didn't", "couldn't", "Could", "wasn't",
                        "won't", "wouldn't", "I have", "I have", "there's", "awesome", "It's", "this", "please", "you",
                        "your", "be", "are", "and", "but", "not", "because", "just", "love", "good", "Good", "well",
                        "great", "Great\\.", "pictures", "picture", "have", "nice", "nice", "Nice", "again", "thanks",
                        "Thanks", "thank you", "excellent", "very", "Very", "favorite", "so", "app", "before", "easy",
                        "wait", "message", "a lot", "lot of", "kind of", "oh, my god", "going to"]

        sent_tokens = nltk.word_tokenize(sent)

        new_sent = ""

        for i, token in enumerate(sent_tokens):
            found_type = False
            for index, lst in enumerate(typos):
                if token in lst:
                    new_sent += replacements[index]
                    if i < (len(sent_tokens)):
                        new_sent += " "
                        found_type = True
                        break

            if not found_type:
                new_sent += token
                if i < (len(sent_tokens)):
                    new_sent += " "

        return new_sent.strip()

    def get_dataset(self, df, content_index, is_eval=False):

        for index, row in df.iterrows():
            if isinstance(row[content_index], str):
                text = row[content_index]
                if is_eval:
                    self.process_dataset(content_index, text, row, is_eval)
                else:
                    sentences = nltk.sent_tokenize(text)
                    for sent in sentences:
                        self.process_dataset(content_index, sent, row, is_eval)

    def breadth_first(self, tree, children=iter):
        """Traverse the nodes of a tree in breadth-first order.
        The first argument should be the tree root; children
        should be a function taking as argument a tree node and
        returning an iterator of the node's children.
        """
        yield tree
        last = tree
        for node in self.breadth_first(tree, children):
            for child in children(node):
                yield child
                last = child
            if last == node:
                return

    def process_dataset(self, content_index, sent, row, is_eval):
        clean_sent = self.clean_sentence(sent)
        clean_sent_words = nltk.word_tokenize(clean_sent)
        sent_pos = nltk.pos_tag(clean_sent_words)

        pos_tags_feature = '-'.join(tag for word, tag in sent_pos)

        sent_str = ''.join([word for word, tag in sent_pos])
        char_bi_gram = [sent_str[i:i + 2] for i in range(len(sent_str) - 2 + 1)]
        char_tri_gram = [sent_str[i:i + 3] for i in range(len(sent_str) - 3 + 1)]
        char_4_gram = [sent_str[i:i + 4] for i in range(len(sent_str) - 4 + 1)]

        all_char_grams = char_bi_gram + char_tri_gram + char_4_gram

        parsed_sent = nlp(clean_sent)

        sent_root = None

        for tok in parsed_sent:
            if tok.dep_ == "ROOT":
                sent_root = tok.orth_
                break

        dep_tree_features = []

        for token in parsed_sent:
            if token.dep_ == 'ROOT':
                dep_tree_features.append(str(token.dep_) + "-" + str(token.tag_))
                root_childs = [str(r.dep_) + "-" + str(r.tag_) for r in token.children]
                dep_tree_features.extend(root_childs)

        output = requests.post(
            url + 'properties={"annotators": "parse", "outputFormat": "json","timeout": "500000"}',
            data={'data': clean_sent}).json()

        result = output['sentences'][0]['parse']
        parsetree = Tree.fromstring(result)
        tree = parsetree[0]
        parse_tree_feature = parsetree.label() + "-"
        i = 1
        for x in self.breadth_first(tree):
            if type(x) != str:
                if x.label() is not None:
                    parse_tree_feature += x.label()
                    i = i + 1
                    if i > 5:
                        break
                    else:
                        parse_tree_feature += "-"

        data_row = {'sentence': clean_sent, 'tagged_sent': sent_pos, 'pos': pos_tags_feature,
                    'char_grams': all_char_grams, 'root': sent_root,
                    'dep_tree': '-'.join(dep_tree_features), 'parse_tree': parse_tree_feature,
                    'class': row[0] if content_index == 1 else None,
                    'reviewId': row[1] if is_eval is False else None,
                    'userName': row[2] if is_eval is False else None,
                    'content': row[4] if is_eval is False else None,
                    'score': row[5] if is_eval is False else None,
                    'thumbsUpCount': row[6] if is_eval is False else None,
                    'reviewCreatedVersion': row[7] if is_eval is False else None,
                    'at': row[8] if is_eval is False else None,
                    'replyContent': row[9] if is_eval is False else None,
                    'repliedAt': row[10] if is_eval is False else None,
                    }

        self.df_dataset = self.df_dataset.append(data_row, ignore_index=True)


if __name__ == '__main__':
    obj = ReviewFeatureExtractor()
    # obj.extract_boc_n_linguistic_features()
    obj.extract_boc_n_linguistic_features_from_app_reviews()
