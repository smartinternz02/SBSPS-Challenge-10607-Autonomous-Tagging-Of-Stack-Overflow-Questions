# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import joblib

# CONVERT Y COLUMN TO CLASSES
multilabel = MultiLabelBinarizer()
tfidf = TfidfVectorizer(analyzer='word', max_features=15, ngram_range=(1, 3), stop_words='english')
sgd = SGDClassifier()
svc = LinearSVC()
labels = ['android', 'asp.net', 'c', 'c#', 'c++', 'css', 'html', 'ios', 'iphone',
        'java', 'javascript', 'jquery', 'mysql', 'objective-c', 'php', 'python', 'ruby',
        'ruby-on-rails', 'sql']

# for classifier in [sgd, svc]:
clf = OneVsRestClassifier(svc)


# Load from file
tagPredictorModel = joblib.load('tagPredictor.pkl')
tfidfModel = joblib.load('tfidf.pkl')

def getTags(question):
    question = tfidfModel.transform(question)
    tags = multilabel.inverse_transform(tagPredictorModel.predict(question))
    print(tags)

ques = 'This is a good question about python and machine learning as well as pandas.'
getTags([ques])