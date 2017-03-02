from tomato import load_train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import cPickle as pickle

'''
svm.py
- This script was used to train and test Support Vector Machine
    models
- The final model and vector used are stored as a tuple in the file,
  models/svm.pkl
- The final model can be extracted using the method, load_svm()
'''

def load_svm(pickle_file = "../models/svm.pkl"):
    '''
    INPUT:
        - pickle_file: file path for pickle file (string)
    OUTPUT:
        - svm_model: sklearn SVC()
        - svm_vector: fitted sklearn CountVectorizer()
    '''
    with open(pickle_file) as f:
        (svm_model, svm_vector) = pickle.load(f)
    return (svm_model, svm_vector)

if __name__ == '__main__':
    (X_train, X_test, y_train, y_test) = load_train_test_split()
    vector = CountVectorizer()
    X_train_vectors = vector.fit_transform(X_train)
    X_test_vectors = vector.transform(X_test)
    model = SVC(kernel="linear").fit(X_train_vectors, y_train)

    print "Train Accuracy:", model.score(X_train_vectors, y_train)
    print "Test Accuracy:", model.score(X_test_vectors, y_test)

    # Train Accuracy: 0.938926479001
    # Test Accuracy: 0.765583083478
