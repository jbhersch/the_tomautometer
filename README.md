# Automating the Tomatometer... The Tomautometer

## Abstract
<a href="http://www.rottentomatoes.com">Rotten Tomatoes</a> is a website that rates
movies using the infamous Tomatometer.  The Tomatometer examines a collection of critic
reviews for any given film and reduces them to either a fresh (positive) or rotten
(negative) rating.  The Tomatometer score is simply the percentage of positive reviews.

For this project, I set out to automate the Tomatometer, which I have cleverly named
the Tomautometer.  The Tomautometer utilizes Natural Language Processing and
Machine Learning to predict the sentiment of language.  The model was trained on
63,367 reviews from 1,172 movies that were scraped from Rotten Tomatoes.  This document
gives an overview of the steps involved to create the Tomautometer.  To see it in action,
check out the <a href="http://ec2-54-144-4-33.compute-1.amazonaws.com:8105/">web app</a>.


## Project Scope
1. Obtain the data
2. Model Development
3. Model Investigation
4. Tomatometer Vs. Tomautomer
5. Web Application

## Obtaining the Data
The data used for this project consists of movie review introductions obtained
from the Rotten Tomatoes website.  The screen shot below shows the review page
for the movie, Snatch.  The introductions to five reviews are shown on the right.
The top three have red tomato icons, indicating they are positive reviews, and the
bottom two have green tomato icons, indicating they are negative reviews.  The structure
of the Rotten Tomatoes review pages allowed for these review introductions to be
scraped and used in the language corpus.  The tomato icons next to each review
allowed for them to very easily labeled good or bad.  The python script,
<a href="https://github.com/jbhersch/the_tomautometer/blob/master/src/rt_scrape.py">rt_scrape.py</a> contains the web scraping code.  

<div style="text-align:center"><img src="images/snatch_screenshot.jpg" width="600" height="400" /></div><br>

## Model Development
Before developing the models the corpus was split into a training set with 80% of the data, and a testing
set with 20% of the data.  My initial plan going into this project was to use an artificial neural network
for the final sentiment model.  While I did create one, it was ultimately not the highest performing model.
Having said that, I spent much more time working on the neural network than the others, so it will be discussed
in more detail.  Another key component in the model development process was language vectorization.  The type of vectorization
varied from model to model, which is discussed below.
#### Models
##### Random Forest
The <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Random Forest </a>
model performed best with 40 trees and the
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html">CountVectorizer</a>
using individual words alone.

The python script, <a href="https://github.com/jbhersch/the_tomautometer/blob/master/src/random_forest.py">random_forest.py</a>, was used for training and testing.
- Train Accuracy: 99.9%
- Test Accuracy: 72.1%

##### Support Vector Machine (SVM)
The <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">SVM</a>
model performed best with the linear kernal function and count vectorization with individual words alone.
As seen below, it beat the Random Forest in test accuracy by roughly 4.5%.

The python script, <a href="https://github.com/jbhersch/the_tomautometer/blob/master/src/svm.py">svm.py</a>, was used for training and testing.
- Train Accuracy: 93.9%
- Test Accuracy: 76.6%

##### Naive Bayes
The <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html">Naive Bayes</a>
model is a multinomial Naive Bayes that uses count vectorization with words and bigrams.  

The python script, <a href="https://github.com/jbhersch/the_tomautometer/blob/master/src/naive_bayes.py">naive_bayes.py</a>, was used for training and testing.
- Train Accuracy: 97.6%
- Test Accuracy: 79.0%

##### Convolutional Neural Network (CNN)
The basis of the CNN model is an example given on
<a href="http://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/">Machine Learning Mastery</a>
used for sentiment analysis on the
<a href="http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf">IMDB Data Set</a>.
One aspect of using neural networks for sentiment analysis is that the language vectorization process is quite different from
the other models used.  In all the other models the scikit-learn CountVectorizer is used with varying hyper parameters.
The CNN uses popularity vectorization, where the vocabulary elements in a review are replaced with high dimensional vectors containing
integers for each element that correspond to its popularity.  In other words, the most frequently
used vocabulary element has a popularity value of one, and so on.  I found that the CNN worked most effectively when generating popularity
vectors from words and bigrams.  Similar to the other models, the CountVectorizer was used to determine the vocabulary list from
the corpus of reviews.  Once the Vectorizer was fit to the corpus, each review in the corpus can be transformed into a count vector,
and then inverse transformed into a list of vocabulary elements. From there, I created a dictionary with vocabulary elements as keys
and their respective popularity as values.  The lists of vocabulary elements corresponding to the reviews were then replaced with
lists of popularity integers obtained from the popularity dictionary.  This process is discussed in more detail in the python script,
<a href="https://github.com/jbhersch/the_tomautometer/blob/master/src/corpus_popularity.py">corpus_popularity.py</a>.

Now that the popularity vectorization process has been discussed, we can get into the structure of the CNN.
On the most basic level, the CNN is a
<a href="https://keras.io/models/sequential/">Keras Sequential model</a>.
The network is constructed with five layers:
- Embedding Layer
- Convolution Layer
- Pooling Layer
- Dense Hidden Layer
- Output Layer

The python script, <a href="https://github.com/jbhersch/the_tomautometer/blob/master/src/cnn.py">cnn.py</a>,
was used for training and testing.
- Train Accuracy: 90.3%
- Test Accuracy: 79.8%

##### Logistic Regression
Surprisingly, the
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Logistic Regression</a>
model outperformed all of the other individual models.  Similar to Naive Bayes and CNN, the Logistic Regression model uses
count vectorization with words and bigrams.

The python script, <a href="https://github.com/jbhersch/the_tomautometer/blob/master/src/logistic_regression.py">logistic_regression.py</a>, was used for training and testing.
- Train Accuracy: 99.9%
- Test Accuracy: 80.0%

##### Ensemble Model
The final model selected is an ensemble model composed of the Logistic Regression model, the Naive Bayes model, and the CNN model.
The predict function of the ensemble model takes an average of the predicted values of its three component models.
While the increase in test accuracy of the ensemble model is not dramatic, it does outperform all the other models
I developed.

The python script, <a href="https://github.com/jbhersch/the_tomautometer/blob/master/src/ensemble.py">ensemble.py</a>, was used for training and testing.
- Train Accuracy: 98.1%
- Test Accuracy: 80.7%

#### Performance
The chart below displays the training and test accuracy for all the models discussed.
From left to right, the models are ordered by increasing test accuracy.  
<div style="text-align:center"><img src="images/model_performance_barchart.png" width="600" height="400" /></div><br>

## Model Investigation
#### Frequency - Sentiment Correlation?
One of the first questions I asked once the final model had been trained is
whether or not there is a correlation between vocabulary frequency and
vocabulary sentiment.  The plot below shows a linear regression on top of a
random sample of 10,000 vocabulary elements plotted against their corresponding
sentiment.  As shown, the R^2 value is minimal which demonstrates that virtually
no correlation exists.

<div style="text-align:center"><img src="images/vocab_sentiment_regression.png" width="600" height="400" /></div><br>

#### Vocabulary Sentiment
<div style="text-align:center"><img src="images/frequency_wordcloud.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/high_sentiment_wordcloud.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/high_sentiment_bigramcloud.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/high_sentiment_barchart.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/low_sentiment_wordcloud.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/low_sentiment_bigramcloud.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/low_sentiment_barchart.png" width="600" height="400" /></div><br>

#### Sentiment Density Functions
<div style="text-align:center"><img src="images/vocab_sentiment_pdf.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/corpus_sentiment_pdf.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/sentiment_pdf.png" width="600" height="400" /></div><br>

## Tomatometer Vs. Tomautometer
<div style="text-align:center"><img src="images/oscar_films_barchart.png" width="600" height="400" /></div><br>

<div style="text-align:center"><img src="images/movie_score_error_pdf.png" width="600" height="400" /></div><br>

## Web Application
- Select a model
- Take in text from a review and score it
- Take in url for a movie on RT, score it, compare it to true score

## Sources
