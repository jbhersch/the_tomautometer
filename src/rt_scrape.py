from bs4 import BeautifulSoup
from requests import get
from tomato import Tomato

'''
rt_scrape.py - module containing methods to scrape the Rotten Tomatoes website
'''

# simple function to convert fresh/rotten ratings
f = lambda x: 1 if x=='fresh' else 0

def get_review_urls(movie_name):
    '''
    INPUT:
        - movie_name: name of movie of interest (string)
    OUTPUT:
        - review_urls: list of all review urls for the movie of interest
    '''
    # set the base review page url for the given movie
    url = "https://www.rottentomatoes.com/m/{0}/reviews/?page=1&sort=".format(movie_name)
    # initiate beautiful soup
    response = get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # get the total number of review pages
    num_pages = int(soup.select('span.pageInfo')[0].text.split()[-1])

    # instantiate the review_urls list
    review_urls = []
    # loop through the number of pages and append each page's url to the list
    for page in xrange(num_pages):
        review_urls.append("https://www.rottentomatoes.com/m/{0}/reviews/?page={1}&sort=".format(movie_name, page+1))
    return review_urls

def get_reviews_and_labels(url):
    '''
    INPUT:
        - url: url for review page of interest (string)
    OUTPUT:
        - reviews: list of reviews shown on given url (string list)
        - labels: list of labels for reviews
    '''

    # initiate beautiful soup
    response = get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # get the html tags for the reviews
    review_tags = soup.select('div.the_review')
    # get the html tags for the review label icons
    icon_tags = soup.select('div.review_icon')
    # instantiate the reviews and labels lists
    reviews = []
    labels = []
    # loop through all reviews and append reviews and labels
    for n in xrange(len(review_tags)):
        if review_tags[n].text != " ":
            reviews.append(review_tags[n].text)
            labels.append(f(str(icon_tags[n]).split('"')[1].split()[-1]))
        # labels.append(str(icon_tags[n]).split('"')[1].split()[-1])
    return reviews, labels

def scrape_movie(movie_name):
    '''
    INPUT:
        - movie_name: name of movie to scrape from rotten tomatoes (string)
    OUTPUT:
        - Tomato object containing movie title, reviews, and labels
    '''
    # get the urls for all the review pages of the movie
    urls = get_review_urls(movie_name)
    # instantiate reviews, and labels lists
    reviews, labels = [], []
    # loop through each review url
    for url in urls:
        # scrape url for reviews and labels
        r, l = get_reviews_and_labels(url)
        # extend reviews and labels lists
        reviews.extend(r)
        labels.extend(l)    
    return Tomato(movie_name, reviews, labels)


if __name__ == '__main__':
    # movie_name = "xxx_return_of_xander_cage"
    # # reviews, labels = scrape_movie(movie_name)
    # xxx = scrape_movie(movie_name)
    #
    # # exec(foo + " = 'something else'")
    # # exec(movie + " = ")
    # d = {}
    # d[xxx.title] = xxx

    url = "https://www.rottentomatoes.com/browse/dvd-all/?services=amazon;amazon_prime;fandango_now;hbo_go;itunes;netflix_iw;vudu#"
    response = get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # info_tags = soup.select('div.movie_info')
    info_tags = soup.select('div#movies-collection')
    # info_tags = soup.findAll('div', attrs={'class': 'movie_info'})
