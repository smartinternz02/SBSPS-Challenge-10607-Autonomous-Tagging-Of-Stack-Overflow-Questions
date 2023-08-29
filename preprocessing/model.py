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
X = tfidf.fit_transform(df['Body'].values.astype(str))
X.shape, y.shape

# SPLITTING DATA INTO TEST AND TRAIN SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
tfidf.vocabulary_

# BUILD MODELS
sgd = SGDClassifier()
lr = LogisticRegression()
svc = LinearSVC()

def j_score(y_true, y_pred):
    # JACCARD SCORE IS USED TO CHECK THE ACCURACY OF A MULTILABEL CLASSIFICATION MODEL
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(axis=1)
    return jaccard.mean() * 100

def print_score(y_pred, clf):
    print("CLF: ", clf.__class__.__name__)
    print("Jaccard score: {}".format(j_score(y_test, y_pred)))
    print("------")

for classifier in [sgd, lr, svc]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_score(y_pred, classifier)

# EXPORTING MODEL
joblib_file = "tagPredictor.pkl"
joblib.dump(clf, joblib_file)

# Load from file
tagPredictorModel = joblib.load('tagPredictor.pkl')

def getTags(question):
    question = tfidf.transform(question)
    tags = multilabel.inverse_transform(tagPredictorModel.predict(question))
    print(tags)


getTags('This is a good question about python and machine learning as well as pandas.')