import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import ast
import requests

df = pd.read_csv("empty.csv")

df["Tags"] = df["Tags"].apply(lambda x: ast.literal_eval(x))
y = df["Tags"]

multilabel = MultiLabelBinarizer()  # multiLabelBinarier for Tags
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

def get_related_tags(tags):
    base_url = "https://api.stackexchange.com/2.3/tags/{}/related".format(";".join(tags))
    params = {
        "site": "stackoverflow"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        related_tags_data = response.json().get("items", [])
        related_tags = [tag.get("name") for tag in related_tags_data]
        return related_tags
    else:
        return []

def suggestion(question):
    x = []
    x.append(question)
    xt = tfidf.transform(x)
    answ = multilabel.inverse_transform(clf.predict(xt))

    suggested_tags = []
    for tags in answ:
        for tag in tags:
            suggested_tags.append(tag)
    
    related_tags = get_related_tags(suggested_tags)
    return suggested_tags, related_tags
