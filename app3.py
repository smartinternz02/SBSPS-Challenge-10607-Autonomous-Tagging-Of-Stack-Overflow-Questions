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
import pickle

# READ DATA
df = pd.read_csv('final-data.csv')
df.head()

# Tag columns is a string. We must convert it to a list.
df['Tag'] = df['Tag'].apply(lambda x: ast.literal_eval(x))
df.head()

# OBTAINING Y AS TARGET VARIABLE
y = df['Tag']

# CONVERT Y COLUMN TO CLASSES
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(y)

# THE CLASSES
multilabel.classes_
pd.DataFrame(y, columns=multilabel.classes_)

# USING TF-IDF VECTORIZER
tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1, 3), stop_words='english')
X = tfidf.fit_transform(df['Title'].values.astype(str))
# X.shape, y.shape

# SPLITTING DATA INTO TEST AND TRAIN SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001, random_state=0)
tfidf.vocabulary_

# BUILD MODELS
sgd = SGDClassifier()
lr = LogisticRegression()
svc = LinearSVC()

# def j_score(y_true, y_pred):
#     # JACCARD SCORE IS USED TO CHECK THE ACCURACY OF A MULTILABEL CLASSIFICATION MODEL
#     jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(axis=1)
#     return jaccard.mean() * 100

# def print_score(y_pred, clf):
#     print("CLF: ", clf.__class__.__name__)
#     print("Jaccard score: {}".format(j_score(y_test, y_pred)))
#     print("------")

for classifier in [svc]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print_score(y_pred, classifier)

import gzip
# sklearn.feature_extraction.text.TfidfVectorizer
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# EXPORTING MODEL
with open('tagPredictor.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load from file
# tagPredictorM=open('tagPredictor.pkl', 'rb')
tagPredictorModel = pickle.load(open('tagPredictor.pkl', 'rb'))
# tfidfM=open('tfidf.pkl', 'rb')
tfidfModel = pickle.load(open('tfidf.pkl', 'rb'))

def getTags(question):
    question = tfidfModel.transform([question])
    tags = multilabel.inverse_transform(tagPredictorModel.predict(question))
    print(tags)
    # print(tfidf.vocabulary_)


# question = 'This is a good question about python and machine learning as well as pandas.'
# tfidf = TfidfVectorizer()
# tfidf.fit_transform([question])

getTags('This is a good question about python and machine learning as well as pandas.')