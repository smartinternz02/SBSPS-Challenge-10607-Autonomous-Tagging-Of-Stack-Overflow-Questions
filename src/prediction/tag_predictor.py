import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# READ DATA
df = pd.read_csv('final-data.csv')
df1 = df.dropna().copy()  # Create a copy of the DataFrame

# Convert Tag column from string to list
df1['Tag'] = df1['Tag'].apply(lambda x: ast.literal_eval(x))

# Load the tagPredictorModel
tagPredictorModel = joblib.load('tagPredictor.pkl')

# Apply TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1, 3), stop_words='english')
X = tfidf.fit_transform(df1['Body'].values.astype(str))

# Apply MultiLabelBinarizer on Tag column
multilabel = MultiLabelBinarizer()
df1['Tag'] = multilabel.fit_transform(df1['Tag'])

def getTags(question):
    question = tfidf.transform(question)
    tags = multilabel.inverse_transform(tagPredictorModel.predict(question))
    return tags


