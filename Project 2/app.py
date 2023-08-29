from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import requests
import ast

app = Flask(__name__)

# Replace with your Stack Overflow API key
API_KEY = "lp8yc4s)kwUbNq54g2WErg(("

# Load your CSV data here
df = pd.read_csv("empty.csv")

df["Tags"] = df["Tags"].apply(lambda x: ast.literal_eval(x))
y = df["Tags"]

multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df["Tags"])
pd.DataFrame(y, columns=multilabel.classes_)

# vectorization
tfidf = TfidfVectorizer(analyzer="word", max_features=10000)
X = tfidf.fit_transform(df["Text"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=10)

# declaration of classifiers
svc = LinearSVC()
clf = OneVsRestClassifier(svc)

clf.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_tags', methods=['POST'])
def get_tags():
    question = request.form['question']
    predicted_tags = suggestion(question)
    
    related_tags = []
    for tag in predicted_tags:
        api_url = f"https://api.stackexchange.com/2.3/tags/{tag}/related?site=stackoverflow&key={API_KEY}"
        response = requests.get(api_url)
        data = response.json()
        related_tags.extend([item['name'] for item in data['items']])
    
    return render_template('index.html', suggested_tags=predicted_tags, related_tags=related_tags)

@app.route('/tags/<tag>/related')
def get_related_tags(tag):
    api_url = f"https://api.stackexchange.com/2.3/tags/{tag}/related?site=stackoverflow&key={API_KEY}"
    
    response = requests.get(api_url)
    data = response.json()
    
    related_tags = [item['name'] for item in data['items']]
    
    return jsonify({"related_tags": related_tags})

def suggestion(question):
    x = []
    x.append(question)
    xt = tfidf.transform(x)
    answ = multilabel.inverse_transform(clf.predict(xt))
    suggested_tags = [tag for tags in answ for tag in tags]
    return suggested_tags

if __name__ == '__main__':
    app.run(debug=True)
