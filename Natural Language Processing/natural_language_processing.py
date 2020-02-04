# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting = 3 itu untuk ignore double quote

# Cleaning the texts
import re #regex
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] #corpus in NLP is collection of text
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #parameter sub 1: what to remove, 2: what to replace the removed, 3: what element to be subbed
    review = review.lower() #to lowercase
    review = review.split() #koyo split js. ini untuk memudahkan stemming dan stopwords
    ps = PorterStemmer()
    #stem is to simplify a word to its root. ex : loved -> love
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #hasilnya itu kata2 yang gak ada di stopwords kek the, that, this, dkk. stopwords itu set
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #only include 1500 words yang sering muncul dan relevant
X = cv.fit_transform(corpus).toarray() #jadi kita punya 1500 words. matrix x cuma 1 sama 0. misal kolom 2 baris 1 valuenya 0. artinya, di review index 0, gaada kata index 2
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)