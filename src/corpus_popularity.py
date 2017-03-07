import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from tomato import Tomato, load_data
import numpy as np

'''
corpus_popularity.py
- This script was used for the popularity vectorization used in the Convolutional
    Neural Network
'''

def popularity_vectorize():
    '''
    INPUT: None
    OUTPUT:
        - vector: CountVectorizier fit to the review corpus
        - pop_dict: Dictionary mapping vocabulary elements to popularity rank
        - pop_list: List of popularity vectors for each review in the corpus
    '''
    # import the review corpus
    movies, corpus, labels = load_data()

    # create a CountVectorizer() instance and fit/transform the corpus
    vector = CountVectorizer(ngram_range = (1,2))
    corpus_vectors = vector.fit_transform(corpus)

    # Calculate the frequency for all elements in the vocabulary
    frequency = np.array(corpus_vectors.sum(axis=0))[0]

    # get the vocab indices in descending order of frequency
    top_indices = np.argsort(-frequency)

    # create list of all vocabualry elements
    feature_names = np.array(vector.get_feature_names())

    # create list of words where order corresponds to popularity
    words = feature_names[top_indices]

    # create the vocab-popularity map dictionary
    pop_dict = {}
    for n, word in enumerate(words): pop_dict[word] = n+1

    # inverse transform the corpus_vectors into lists of vocab elements
    corpus_words = vector.inverse_transform(corpus_vectors)

    # create the list of popularity vectors for each review in the corpus
    # using the popularity dictionary
    pop_list = []
    for review in corpus_words:
        review_nums = []
        for word in review:
            review_nums.append(pop_dict[word])
        pop_list.append(review_nums)
    return (vector, pop_dict, pop_list)

if __name__ == '__main__':
    vector, pop_dict, pop_list = popularity_vectorize()
