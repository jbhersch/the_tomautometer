from tomato import load_train_test_split, Tomato
import cPickle as pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

'''
logistic_regression.py
- This script was used to train and test Logistic Regression models
- The final model and vector used are stored as a tuple in the file,
  models/logistic_regression.pkl
- The final model can be extracted using the method, load_logistic_regression()
'''

def load_logistic_regression(pickle_file = "../models/logistic_regression.pkl"):
    '''
    INPUT:
        - pickle_file: file path for pickle file (string)
    OUTPUT:
        - logistic_regression_model: sklearn LogisticRegression()
        - logistic_regression_vector: fitted sklearn CountVectorizer()
    '''
    with open(pickle_file) as f:
        (logistic_regression_model, logistic_regression_vector) = pickle.load(f)
    return (logistic_regression_model, logistic_regression_vector)

if __name__ == '__main__':
    # load the training/testing data
    (X_train, X_test, y_train, y_test) = load_train_test_split()
    # instantiate the vectorizer
    vector = CountVectorizer(ngram_range=(1,2))
    # fit the vectorizer and transform the reviews into vectors
    X_train_vectors = vector.fit_transform(X_train)
    X_test_vectors = vector.transform(X_test)
    # fit the model
    model = LogisticRegression(fit_intercept=False).fit(X_train_vectors, y_train)
    # output results
    print "Train Accuracy:", model.score(X_train_vectors, y_train)
    print "Test Accuracy:", model.score(X_test_vectors, y_test)
    # Train Accuracy: 0.998993943937
    # Test Accuracy: 0.800142023039
