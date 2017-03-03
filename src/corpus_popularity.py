import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from tomato import Tomato, load_data
import numpy as np

def popularity_vectorize():
    movies, corpus, labels = load_data()
    vector = CountVectorizer(ngram_range = (1,2))
    corpus_vectors = vector.fit_transform(corpus)
    frequency = np.array(corpus_vectors.sum(axis=0))[0]
    top_indices = np.argsort(-frequency)
    feature_names = np.array(vector.get_feature_names())
    words = feature_names[top_indices]
    pop_dict = {}
    for n, word in enumerate(words): pop_dict[word] = n+1
    corpus_words = vector.inverse_transform(corpus_vectors)
    pop_list = []
    for review in corpus_words:
        review_nums = []
        for word in review:
            review_nums.append(pop_dict[word])
        pop_list.append(review_nums)
    return (vector, pop_dict, pop_list)

if __name__ == '__main__':
    vector, pop_dict, pop_list = popularity_vectorize()
