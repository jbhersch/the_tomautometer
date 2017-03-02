from tomato import load_train_test_split, Tomato
import cPickle as pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

'''
naive_bayes.py
- This script was used to train and test Naive Bayes models.
- The final model and vector used are stored as a tuple in the file,
  models/naive_bayes.pkl
- The final model can be extracted using the method, load_naive_bayes()
'''

def load_naive_bayes(pickle_file = "../models/naive_bayes.pkl"):
    '''
    INPUT:
        - pickle_file: file path for pickle file (string)
    OUTPUT:
        - naive_bayes_model: sklearn MultinomialNB()
        - naive_bayes_vector: fitted sklearn CountVectorizer()
    '''
    with open(pickle_file) as f:
        (naive_bayes_model, naive_bayes_vector) = pickle.load(f)
    return (naive_bayes_model, naive_bayes_vector)

if __name__ == '__main__':
    (X_train, X_test, y_train, y_test) = load_train_test_split()
    vector = CountVectorizer(ngram_range=(1,2))
    X_train_vectors = vector.fit_transform(X_train)
    X_test_vectors = vector.transform(X_test)
    model = MultinomialNB().fit(X_train_vectors, y_train)

    print "Train Accuracy:", model.score(X_train_vectors, y_train)
    print "Test Accuracy:", model.score(X_test_vectors, y_test)
    # Train Accuracy: 0.976347819226
    # Test Accuracy: 0.789963705223
