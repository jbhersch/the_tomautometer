from src.ensemble import load_ensemble_model
from src.tomato import load_train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    logistic_regression_file = "models/logistic_regression.pkl"
    naive_bayes_file = "models/naive_bayes.pkl"
    cnn_files = ("models/cnn.pkl",
                 "models/cnn.json",
                 "models/cnn_weights.h5")
    train_test_file = "data/train_test.pkl"
    ensemble_model = load_ensemble_model(logistic_regression_file,
                                         naive_bayes_file,
                                         cnn_files)
    # X = ["good movie", "bad movie"]
    # print ensemble_model.predict(X)
    # print ensemble_model.predict(X, True)
    pos_grams = ["good", "really good", "very good", "excellent", "amazing", "awesome", "great"]
    med_grams = ["okay", "decent", "average", "mediocre"]
    neg_grams = ["not good", "bad", "very bad", "terrible", "awful"]

    pos_grams = [gram for gram in pos_grams if gram in ensemble_model.cnn_pop_dict]
    med_grams = [gram for gram in med_grams if gram in ensemble_model.cnn_pop_dict]
    neg_grams = [gram for gram in neg_grams if gram in ensemble_model.cnn_pop_dict]

    pos_pop = [ensemble_model.cnn_pop_dict[gram] for gram in pos_grams]
    med_pop = [ensemble_model.cnn_pop_dict[gram] for gram in med_grams]
    neg_pop = [ensemble_model.cnn_pop_dict[gram] for gram in neg_grams]

    pos_prob = ensemble_model.predict(pos_grams, True)
    med_prob = ensemble_model.predict(med_grams, True)
    neg_prob = ensemble_model.predict(neg_grams, True)

    pos = sorted(zip(pos_grams, pos_pop), key = lambda x: x[1])
    med = sorted(zip(med_grams, med_pop), key = lambda x: x[1])
    neg = sorted(zip(neg_grams, neg_pop), key = lambda x: x[1])

    
