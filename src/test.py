from tomato import load_data, Tomato, load_train_test_split
import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split
from logistic_regression import load_logistic_regression
from naive_bayes import load_naive_bayes
from cnn import load_cnn, cap_top_words, popularity_vector
from keras.preprocessing import sequence
from svm import load_svm
from random_forest import load_random_forest

if __name__ == '__main__':
    (X_train, X_test, y_train, y_test) = load_train_test_split()

    # Logistic Regression
    (logistic_regression_model, logistic_regression_vector) = load_logistic_regression()
    logistic_regression_train = logistic_regression_vector.transform(X_train)
    logistic_regression_test = logistic_regression_vector.transform(X_test)
    logistic_regression_train_accuracy = logistic_regression_model.score(logistic_regression_train, y_train)
    logistic_regression_test_accuracy = logistic_regression_model.score(logistic_regression_test, y_test)

    # Naive Bayes
    (naive_bayes_model, naive_bayes_vector) = load_naive_bayes()
    naive_bayes_train = naive_bayes_vector.transform(X_train)
    naive_bayes_test = naive_bayes_vector.transform(X_test)
    naive_bayes_train_accuracy = naive_bayes_model.score(naive_bayes_train, y_train)
    naive_bayes_test_accuracy = naive_bayes_model.score(naive_bayes_test, y_test)

    # CNN
    (cnn_model, cnn_vector, cnn_pop_dict, cnn_word_cap, cnn_pad_words) = load_cnn()
    cnn_train = popularity_vector(X_train, cnn_vector, cnn_pop_dict)
    cnn_test = popularity_vector(X_test, cnn_vector, cnn_pop_dict)
    cnn_train = cap_top_words(cnn_train, cnn_word_cap)
    cnn_test = cap_top_words(cnn_test, cnn_word_cap)
    cnn_train = sequence.pad_sequences(np.array(cnn_train), maxlen=cnn_pad_words)
    cnn_test = sequence.pad_sequences(np.array(cnn_test), maxlen=cnn_pad_words)
    cnn_train_accuracy = cnn_model.evaluate(cnn_train, y_train, verbose=0)[1]
    cnn_test_accuracy = cnn_model.evaluate(cnn_test, y_test, verbose=0)[1]

    # SVM
    (svm_model, svm_vector) = load_svm()
    svm_train = svm_vector.transform(X_train)
    svm_test = svm_vector.transform(X_test)
    svm_train_accuracy = svm_model.score(svm_train, y_train)
    svm_test_accuracy = svm_model.score(svm_test, y_test)

    # Random Forest
    (random_forest_model, random_forest_vector) = load_random_forest()
    random_forest_train = random_forest_vector.transform(X_train)
    random_forest_test = random_forest_vector.transform(X_test)
    random_forest_train_accuracy = random_forest_model.score(random_forest_train, y_train)
    random_forest_test_accuracy = random_forest_model.score(random_forest_test, y_test)
