import nltk
from sklearn.metrics.pairwise import cosine_similarity


def cos_sim(sent_vec_1, sent_vec_2):
    sim = cosine_similarity([sent_vec_1, sent_vec_2])
    return sim.tolist()


def lemmatize_stemming(text):
    stemmer = nltk.SnowballStemmer('english')
    return stemmer.stem(nltk.WordNetLemmatizer().lemmatize(text, pos='v'))


def get_jaccard_sim(str1, str2):
    l1 = lemmatize_stemming(str1)
    l2 = lemmatize_stemming(str2)
    a = set(l1.split())
    b = set(l2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
