from bs4 import BeautifulSoup
from requests import get
from tomato import Tomato

'''
rt_scrape.py - module containing methods to scrape the Rotten Tomatoes website
'''

# simple function to convert fresh/rotten ratings
f = lambda x: 1 if x=='fresh' else 0

def get_review_urls(movie_name):
    url = "https://www.rottentomatoes.com/m/{0}/reviews/?page=1&sort=".format(movie_name)
    response = get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    num_pages = int(soup.select('span.pageInfo')[0].text.split()[-1])

    review_urls = []
    for page in xrange(num_pages):
        review_urls.append("https://www.rottentomatoes.com/m/{0}/reviews/?page={1}&sort=".format(movie_name, page+1))
    return review_urls

def get_reviews_and_labels(url):
    response = get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    review_tags = soup.select('div.the_review')
    icon_tags = soup.select('div.review_icon')
    reviews = []
    labels = []
    for n in xrange(len(review_tags)):
        if review_tags[n].text != " ":
            reviews.append(review_tags[n].text)
            labels.append(f(str(icon_tags[n]).split('"')[1].split()[-1]))
        # labels.append(str(icon_tags[n]).split('"')[1].split()[-1])
    return reviews, labels

def scrape_movie(movie_name):
    urls = get_review_urls(movie_name)
    reviews, labels = [], []
    for url in urls:
        r, l = get_reviews_and_labels(url)
        reviews.extend(r)
        labels.extend(l)
    # return reviews, labels
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
