import numpy as np
from logistic_regression import load_logistic_regression
from naive_bayes import load_naive_bayes
from cnn import load_cnn, cap_top_words, popularity_vector
from keras.preprocessing import sequence
from tomato import load_train_test_split, Tomato

class Ensemble(object):
    '''
    The Ensemble class is an ensemble of three different models - Logistic
    Regression, Naive Bayes, Convulutional Neural Network (CNN).  The class
    imports the necessary objects to perform sentiment prediction for each model
    upon instantiation.  Similar to other machine learning models, this
    Ensemble class has methods for predicting on new data and calculating
    accuracy for data with sentiment that is already known.

    ATTRIBUTES:
        - logistic_regression_model: sklearn LogisticRegression() object
        - logistic_regression_vector: sklearn CountVectorizer() required
            by the LogisticRegression() object
        - naive_bayes_model: sklearn MultinomialNB() object
        - naive_bayes_vector: sklearn CountVectorizer() required by the
            MultinomialNB() object
        - cnn_model: keras sequential CNN object
        - cnn_vector: sklearn CountVectorizer()
        - cnn_pop_dict: Dictionary mapping all words in the corpus to their
            popularity (dict)
        - cnn_word_cap: Maximum allowed popuarity (int)
        - cnn_pad_words: Number of words to pad each popularity vector (int)
    '''
    def __init__(self, logistic_regression, naive_bayes, cnn):
        # Logistic Regression
        self.logistic_regression_model = logistic_regression[0]
        self.logistic_regression_vector = logistic_regression[1]
        # Naive Bayes
        self.naive_bayes_model = naive_bayes[0]
        self.naive_bayes_vector = naive_bayes[1]
        # CNN
        self.cnn_model = cnn[0]
        self.cnn_vector = cnn[1]
        self.cnn_pop_dict = cnn[2]
        self.cnn_word_cap = cnn[3]
        self.cnn_pad_words = cnn[4]

    def logistic_regression_predict(self, X, predict_proba = False):
        '''
        INPUT:
            - X: List of documents to predict on (string list)
        OUTPUT:
            - Binary predictions on X (ndarray)
        '''
        X_vectors = self.logistic_regression_vector.transform(X)
        if predict_proba:
            return self.logistic_regression_model.predict_proba(X_vectors)[:,1]
        else:
            return self.logistic_regression_model.predict(X_vectors)


    def naive_bayes_predict(self, X, predict_proba = False):
        '''
        INPUT:
            - X: List of documents to predict on (string list)
        OUTPUT:
            - Binary predictions on X (ndarray)
        '''
        X_vectors = self.naive_bayes_vector.transform(X)
        if predict_proba:
            return self.naive_bayes_model.predict_proba(X_vectors)[:,1]
        else:
            return self.naive_bayes_model.predict(X_vectors)

    def cnn_predict(self, X, predict_proba = False):
        '''
        INPUT:
            - X: List of documents to predict on (string list)
        OUTPUT:
            - Binary predictions on X (ndarray)
        '''
        X_vectors = popularity_vector(X, self.cnn_vector, self.cnn_pop_dict)
        X_vectors = cap_top_words(X_vectors, self.cnn_word_cap)
        X_vectors = sequence.pad_sequences(np.array(X_vectors), maxlen=self.cnn_pad_words)
        if predict_proba:
            return self.cnn_model.predict(X_vectors, verbose=0)[:,0]
        else:
            return self.cnn_model.predict_classes(X_vectors, verbose=0)[:,0]

    def predict(self, X, predict_proba = False):
        '''
        INPUT:
            - X: List of documents to predict on (string list)
        OUTPUT:
            - Binary predictions on X (ndarray)
        '''
        if predict_proba:
            pred = np.vstack((self.logistic_regression_predict(X, True),
                              self.naive_bayes_predict(X, True),
                              self.cnn_predict(X, True))).T
            return np.mean(pred, axis=1)
        else:
            pred = np.vstack((self.logistic_regression_predict(X),
                              self.naive_bayes_predict(X),
                              self.cnn_predict(X))).T
            return np.round(np.mean(pred, axis=1))

    def predict_tomato(self, movie, predict_proba = False):
        '''
        INPUT:
            - movie: Tomato object containing reviews scraped from rotten
                tomatoes for the given movie.
            - predict_proba: Whether or not to aggregate the classification
                predictions or probability predictions (boolean)
        OUTPUT:
            - Tomautometer score (float)
        '''
        return np.mean(self.predict(movie.reviews, predict_proba))

    def score(self, X, y):
        '''
        INPUT:
            - X: List of documents to predict on (string list)
            - y: List of binary labels to predict against (int list)
        OUTPUT:
            - Accuracy of predictions on X against y (float)
        '''
        return np.mean(self.predict(X)==y)


def load_ensemble_model(logistic_regression_file = "../models/logistic_regression.pkl",
                        naive_bayes_file = "../models/naive_bayes.pkl",
                        cnn_files = ("../models/cnn.pkl", "../models/cnn.json", "../models/cnn_weights.h5")):
    '''
    INPUT:
        - logistic_regression_file: file path of logistic regression file (str)
        - naive_bayes_file: file path of naive bayes file (str)
        - cnn_files: 3 dimensional tuple of cnn files (str tuple)
    OUTPUT:
        - Ensemble object
    '''
    return Ensemble(load_logistic_regression(logistic_regression_file),
                    load_naive_bayes(naive_bayes_file),
                    load_cnn(cnn_files[0], cnn_files[1], cnn_files[2]))


if __name__ == '__main__':
    (X_train, X_test, y_train, y_test) = load_train_test_split()
    ensemble = load_ensemble_model()
    print "Train Accuracy:", ensemble.score(X_train, y_train)
    print "Test Accuracy:", ensemble.score(X_test, y_test)
