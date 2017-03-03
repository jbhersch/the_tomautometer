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
using individual words alone.  The Train/Test accuracies are listed below.
- Train Accuracy: 99.9%
- Test Accuracy: 72.1%

##### Support Vector Machine (SVM)
The <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">SVM</a>
model performed best with the linear kernal function and count vectorization with individual words alone.
As seen below, it beat the Random Forest in test accuracy by roughly 4.5%.
- Train Accuracy: 93.9%
- Test Accuracy: 76.6%

##### Naive Bayes
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html">Naive Bayes</a>
- Train Accuracy: 97.6%
- Test Accuracy: 79.0%

##### Convolutional Neural Network (CNN)
- Train Accuracy: 90.3%
- Test Accuracy: 79.8%

##### Logistic Regression
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Logistic Regression</a>
- Train Accuracy: 99.9%
- Test Accuracy: 80.0%

##### Ensemble Model
- Train Accuracy: 98.1%
- Test Accuracy: 80.7%

#### Performance
The chart below displays the training and test accuracy for all the models discussed.
From left to right, the models are ordered by increasing test accuracy.  As shown,
the Random Forest and SVM were the two weakest models in terms of accuracy, both
of which have test accuracies in the low to mid 70th percentile.  Naive Bayes and
CNN have test accuracies just under 80% and Logistic Regression is just above 80%.
Finally, the most accurate model, the ensemble model, is slightly better than
Logistic Regression with a test accuracy just under 81%.  
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
