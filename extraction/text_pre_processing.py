import html
import re

import nltk
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode

stemmer = SnowballStemmer("english")


class TextProcessing:
    def __init__(self, app_id, data):
        self.appId = app_id
        self.data = data

    def segment_into_sentences(self, sents_already_segmented=False):
        self.sents = []
        self.data = unidecode(self.data)
        pattern = r'".*"(\s+-.*)?'
        self.data = re.sub(pattern, '', self.data)
        pattern1 = r'\'s'
        self.data = re.sub(pattern1, "", self.data)

        if sents_already_segmented:
            list_lines = self.data.split('\n')
            list_lines = [line for line in list_lines if line.strip() != '']
            for line in list_lines:
                if line.strip() == "Credits" or line.strip() == "credits":
                    break

                if line.isupper():
                    line = line.capitalize()

                sentences = nltk.sent_tokenize(line)
                sentences = [self.replace_common_typos(sent) for sent in sentences if sent.strip() != '']
                self.sents.extend(sentences)
        elif not sents_already_segmented:
            self.sents = nltk.sent_tokenize(self.data)
            self.sents = [sent for sent in self.sents if sent.strip() != '']
            self.sents = [self.replace_common_typos(sent) for sent in self.sents if sent.strip() != '']

        return self.sents

    # clean sentences
    @staticmethod
    def replace_common_typos(sent):
        typos = [["im", "i m", "I m", "I'm", "i'm"], ["Ill", "I'll", "i'll"], ["cnt", "cant", "can't", "cann't"],
                 ["doesnt", "doesn't"], ["dont", "don't"], ["isnt", "isn't"], ["didnt", "didn't"],
                 ["couldnt", "couldn't"], ["Cud"], ["wasnt", "wasn't"], ["wont", "won't"], ["wouldnt", "wouldn't"],
                 ["ive", "i've"], ["Ive", "I've"], ["theres", "there's"], ["awsome", "awsum", "awsm"],
                 ["Its", "it's", 'itz', "it'z"], ["dis", "diz"], ["plzz", "Plz ", "plz", "pls ", "Pls", "Pls "],
                 ["U", "u"], ["ur"], ["b"], ["r"], ["nd ", "n", "&"], ["bt"], ["nt"], ["coz", "cuz"], ["jus", "jst"],
                 ["luv", "Luv"], ["gud"], ["Gud"], ["wel"], ["gr8", "Gr8", "Grt", "grt"], ["Gr\\."], ["pics"], ["pic"],
                 ["hav"], ["nic"], ["nyc", "9ce"], ["Nic"], ["agn"], ["thx", "tks", "thanx"], ["Thx"], ["thkq"],
                 ["Xcellent"], ["vry"], ["Vry"], ["fav", "favo"], ["soo", "sooo", "soooo"], ["ap"], ["b4"], ["ez"],
                 ["w8"], ["msg"], ["alot"], ["lota"], ["kinda"], ["omg"], ["gota"], ['v', 'w', 'V', 'W'], ['c'], ['b'],
                 ['nt']]
        replacements = ["I am", "i will", "can not", "does not", "do not", "is not", "did not", "could not", "Could",
                        "was not", "will not", "would not", "I have", "I have", "there is", "awesome", "It is", "this",
                        "please", "you", "your", "be", "are", "and", "but", "not", "because", "just", "love", "good",
                        "Good", "well", "great", "Great\\.", "pictures", "picture", "have", "nice", "nice", "Nice",
                        "again", "thanks", "Thanks", "thank you", "excellent", "very", "Very", "favorite", "so", "app",
                        "before", "easy", "wait", "message", "a lot", "lot of", "kind of", "oh, my god", "going to",
                        "we", "see", "be", "not"]

        sent_tokens = nltk.word_tokenize(sent)

        new_sent = ""

        for i, token in enumerate(sent_tokens):
            found_type = False
            for index, lst in enumerate(typos):
                if token in lst:
                    new_sent += replacements[index]
                    if i < (len(sent_tokens) - 1):
                        new_sent += " "
                        found_type = True
                        break

            if not found_type:
                new_sent += token
                if i < (len(sent_tokens) - 1):
                    new_sent += " "

        return new_sent.strip()

    @staticmethod
    def get_opinion_lexicon():
        opinion_lexicon_path = '../data/stop_words'

        with open(opinion_lexicon_path, 'r') as f:
            content = f.readlines()

        opinion_lexicon = [x.strip() for x in content]

        return opinion_lexicon

    def get_clean_sentences(self):
        sentences = []
        clean_sentences = []
        # remove explanations text with-in brackets
        for sent in self.sents:
            sent = html.unescape(sent.strip())
            sent = sent.lstrip('-')
            regex = r"(\(|\[).*?(\)|\])"
            urls = re.findall(
                '(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?',
                sent)
            emails = re.findall(
                "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*",
                sent)
            match_list = re.finditer(regex, sent)

            new_sent = sent

            if len(urls) == 0 and len(emails) == 0:
                if match_list:
                    for match in match_list:
                        txt_to_be_removed = sent[match.start():match.end()]
                        new_sent = new_sent.replace(txt_to_be_removed, "")

                    clean_sentences.append(new_sent)
                else:
                    clean_sentences.append(sent)

        pattern = r'\*|\u2022|#'

        custom_stop_words = set(self.get_opinion_lexicon())
        cutom_stop_words_stemmed = [stemmer.stem(w) for w in custom_stop_words]

        for index, sent in enumerate(clean_sentences):
            clean_sent = re.sub(pattern, "", sent)
            # removing sub-ordinate clauses from a sentence
            sent_wo_clause = self.remove_sub_ordinate_clause(clean_sent)

            clean_sentences[index] = sent_wo_clause

            tokens = nltk.word_tokenize(clean_sentences[index])

            sent_tokens = [w.lower() for w in tokens if stemmer.stem(w.lower()) not in cutom_stop_words_stemmed]

            sentences.append(' '.join(sent_tokens))

        return sentences

    @staticmethod
    def remove_sub_ordinate_clause(sentence):
        sub_ordinate_words = ['when', 'after', 'although', 'because', 'before', 'if', 'rather', 'since', 'though',
                              'unless', 'until', 'whenever', 'where', 'whereas', 'wherever', 'whether', 'while', 'why',
                              'which', 'by', 'so', 'but'
                              ]

        sub_ordinate_clause = False
        words = []
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token.lower() in sub_ordinate_words:
                sub_ordinate_clause = True

            if not sub_ordinate_clause:
                words.append(token)

            elif sub_ordinate_clause:
                break

        return ' '.join(words).strip()
