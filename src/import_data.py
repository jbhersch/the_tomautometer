import rt_scrape as scrape
from tomato import Tomato
import pandas as pd
import sys

'''
import_data.py - script that scrapes the rotten tomatoes site and stores the data.
'''

if __name__ == '__main__':
    # import csv file containing list of all movie urls into a pandas datafram
    df = pd.read_csv("../data/movie_list.csv")
    # strip the movie titles from the urls and put them in a list
    titles = [x.split('/')[-1] for x in df.values[:,0]]
    # remove any duplicates
    titles = list(set(titles))
    # check for broken links and remove if they exist
    if 'null' in titles:
        titles.pop(titles.index('null'))

    tomatoes = {} # dictionary of Tomato objects
    errors = {} # dictionary of import errors
    corpus = [] # list of all reviews
    labels = [] # list of all labels
    n = 0 # counting variable to watch progress
    for title in titles:
        n += 1
        try:
            tomatoes[title] = scrape.scrape_movie(title)
            corpus.extend(tomatoes[title].reviews)
            labels.extend(tomatoes[title].labels)
        except:
            errors[title] = sys.exc_info()[0]
        print n
