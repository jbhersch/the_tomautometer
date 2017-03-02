from tomato import load_train_test_split
import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

'''
random_forest.py
- This script was used to train and test Random Forest models
- The final model and vector used are stored as a tuple in the file,
  models/random_forest.pkl
- The final model can be extracted using the method, load_random_forest()
'''

def load_random_forest(pickle_file = "../models/random_forest.pkl"):
    '''
    INPUT:
        - pickle_file: file path for pickle file (string)
    OUTPUT:
        - random_forest_model: sklearn RandomForestClassifier()
        - random_forest_vector: fitted sklearn CountVectorizer()
    '''
    with open(pickle_file) as f:
        (random_forest_model, random_forest_vector) = pickle.load(f)
    return (random_forest_model, random_forest_vector)

if __name__ == '__main__':
    (X_train, X_test, y_train, y_test) = load_train_test_split()

    vector = CountVectorizer(ngram_range=(1,1))
    X_train_vectors = vector.fit_transform(X_train)
    X_test_vectors = vector.transform(X_test)

    model = RandomForestClassifier(n_estimators=40)
    model.fit(X_train_vectors, y_train)

    print "Train Accuracy:", model.score(X_train_vectors, y_train)
    print "Test Accuracy:", model.score(X_test_vectors, y_test)

    # 40 trees, no bigrams
    # Train Accuracy: 0.999802734105
    # Test Accuracy: 0.72139813792
