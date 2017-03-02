from tomato import load_train_test_split
import numpy as np
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier


if __name__ == '__main__':
    (X_train, X_test, y_train, y_test) = load_train_test_split()

    vector = CountVectorizer(ngram_range=(1,1))
    X_train_vectors = vector.fit_transform(X_train)
    X_test_vectors = vector.transform(X_test)

    model = GradientBoostingClassifier()
    model.fit(X_train_vectors, y_train)

    print "Train Accuracy:", model.score(X_train_vectors, y_train)
    print "Test Accuracy:", model.score(X_test_vectors, y_test)
