from tomato import load_train_test_split, Tomato
import cPickle as pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
from sklearn.feature_extraction.text import CountVectorizer

'''
cnn.py
- This script was used to train and test CNN models.
- The final model, and it's weights are stored in the files cnn.json and
    cnn_weights.h5
- The vectorizer, popuarity dictionary, word cap, and number of pad words are
    stored in cnn.pkl
- The final model and the other objects required for it to run can be extracted
    using load_cnn()
'''

def popularity_vector(X, vector, pop_dict):
    '''
    INPUT:
        - X: Reviews to be formatted as popuarity vector (string list)
        - vector: Vectorizer that transforms strings into lists of the words
        - pop_dict: Map of each word in corpus to popularity integer (dict)
    OUTPUT:
        - pop_vector: Popularity vectors for reviews in X (list of int lists)
    '''
    X_words = vector.inverse_transform(vector.transform(X))
    pop_vector = []
    for words in X_words:
        words_vector = []
        for word in words:
            words_vector.append(pop_dict[word])
        pop_vector.append(words_vector)
    return pop_vector

def cap_top_words(pop_list, word_cap, word_floor=2):
    '''
    INPUT:
        - pop_list: List of popularity vectors (int list)
        - word_cap: Highest popularity integer to be allowed (int)
        - word_floor: Integer that popularity values greater than word_cap to
            be replaced with (int)
    OUTPUT:
        - cap_list: Popularity vectors with values higher than word_cap replaced
            with word_floor (int list)
    '''
    cap_list = []
    for lst in pop_list:
        cap = []
        for num in lst:
            if num > word_cap - 1:
                cap.append(word_floor)
            else:
                cap.append(num)
        cap_list.append(cap)
    return cap_list

def load_cnn(cnn_file = "../models/cnn.pkl", json_file = "../models/cnn.json", h5_file = "../models/cnn_weights.h5"):
    '''
    INPUT:
        - cnn_file: pickle file path containing vectorizer, popularity dictionary,
            word cap, and number of pad words (string)
        - json_file: json file path containing the cnn model information (string)
        - h5_file: h5 file path containing the cnn model weights (string)
    OUTPUT:
        - cnn_model: keras sequential() cnn model
        - cnn_vector: fitted sklearn CountVectorizer()
        - cnn_pop_dict: Dictionary mapping words to their popularity (dict)
        - cnn_word_cap: Highest popularity used in model (int)
        - cnn_pad_words: Number of elements in vectorized data (int)
    '''
    with open(cnn_file) as f:
        (cnn_vector, cnn_pop_dict, cnn_word_cap, cnn_pad_words) = pickle.load(f)
    open_json = open(json_file,"r")
    cnn_model_json = open_json.read()
    open_json.close()
    cnn_model = model_from_json(cnn_model_json)
    cnn_model.load_weights(h5_file)
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return (cnn_model, cnn_vector, cnn_pop_dict, cnn_word_cap, cnn_pad_words)


if __name__ == '__main__':
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    (X_train, X_test, y_train, y_test) = load_train_test_split()
    with open("../data/corpus_pop_bigrams.pkl") as f:
        vector, pop_dict, pop_list = pickle.load(f)
    sample_size = 10000
    X_train_vectors = popularity_vector(X_train, vector, pop_dict)
    X_test_vectors = popularity_vector(X_test, vector, pop_dict)

    word_cap = 200000
    pad_words = max([len(x) for x in X_train_vectors + X_test_vectors])

    X_train_vectors = cap_top_words(X_train_vectors, word_cap)
    X_test_vectors = cap_top_words(X_test_vectors, word_cap)

    X_train_vectors = sequence.pad_sequences(np.array(X_train_vectors), maxlen=pad_words)
    X_test_vectors = sequence.pad_sequences(np.array(X_test_vectors), maxlen=pad_words)
    y_train, y_test = np.array(y_train), np.array(y_test)

    model = Sequential()
    model.add(Embedding(word_cap, 32, input_length=pad_words))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='softplus'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='softplus'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train_vectors, y_train, validation_data=(X_test_vectors, y_test), nb_epoch=2, batch_size=128, verbose=1)

    scores = model.evaluate(X_test_vectors, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # train accuracy: 0.90266900755763535
    # test accuracy: 0.79824838251538588
