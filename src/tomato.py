import cPickle as pickle
import numpy as np

'''
tomato.py
'''

class Tomato(object):
    '''
    The Tomato class stores information scraped from Rotten Tomatoes
    for a single movie.

    ATTRIBUTES:
        - title: title of the movie. (string)
        - reviews: list of movie reviews. (string list)
        - labels: list of binary values corresponding positive (1) or negative (0) reviews (int list)
        - count: number of reviews or labels (int)
        - score: percentage of good reviews (float)
    '''
    def __init__(self, title, reviews, labels):
        self.title = title
        self.reviews = reviews
        self.labels = labels
        self.count = len(reviews)
        self.score = 1.*sum(labels)/self.count

def load_data(tomato_file = "../data/tomatoes.pkl", corpus_file = "../data/corpus.pkl", labels_file = "../data/labels.pkl"):
    '''
    INPUT:
        - tomato_file: path to pickle file containing tomato dict (string)
        - corpus_file: path to pickle file containing corpus (string)
        - labels_file: path to pickle file containing labels (string)
    OUTPUT:
        - tomatoes: dictionary of Tomato objects (Tomato dict)
        - reviews: list of all movie reviews (string list)
        - labels: list of all movie labels (int list)
    '''
    with open(tomato_file) as f:
        tomatoes = pickle.load(f)
    with open(corpus_file) as f:
        corpus = pickle.load(f)
    with open(labels_file) as f:
        labels = pickle.load(f)

    return tomatoes, corpus, labels

def load_train_test_split(pickle_file = "../data/train_test.pkl"):
    '''
    INPUT:
        - pickle_file: path to pickle file containing data split (string)
    OUTPUT:
        - X_train: Training set of movie reviews (string list)
        - X_test: Testing set of movie reviews (string list)
        - y_train: Training set of movie labels (int list)
        - y_test: Testing set of movie labels (int list)
    '''

    with open(pickle_file) as f:
        (X_train, X_test, y_train, y_test) = pickle.load(f)
    return (X_train, X_test, y_train, y_test)

def subsample_data(corpus, labels):
    corpus, labels = np.array(corpus), np.array(labels)
    pos_indices = np.where(labels==1)[0]
    neg_indices = np.where(labels==0)[0]
    pos_keep = np.random.choice(pos_indices, size=len(neg_indices), replace=False)
    balanced_corpus = np.hstack((corpus[pos_keep], corpus[neg_indices]))
    balanced_labels = np.hstack((labels[pos_keep], labels[neg_indices]))
    return (balanced_corpus, balanced_labels)

if __name__ == '__main__':
    tomatoes, corpus, labels = load_data()
