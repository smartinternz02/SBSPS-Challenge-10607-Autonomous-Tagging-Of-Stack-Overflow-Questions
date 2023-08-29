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
# y_pred = clf.predict(X_test)
# print(y_pred)
# pickled_model = pickle.load(open('model.pkl', 'rb'))


# predicting the model with samples

def suggestion(question):
    x=[]
    x.append(question)
    xt=tfidf.transform(x)
    answ = multilabel.inverse_transform(clf.predict(xt))

    suggested_tags = []
    for tags in answ:
        for tag in tags:
            suggested_tags.append(tag)
    
    return suggested_tags


def main():
    # Get the question from the user
    question = input("Enter your question: ")

    # Predict the tags for the question
    predicted_tags = suggestion(question)

    # Print the predicted tags
    print("Suggested tags:", predicted_tags)


# Function to search Stack Overflow and get relevant question links
def search_stack_overflow(tags):
    base_url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "order": "desc",
        "sort": "votes",
        "q": " ".join(tags),
        "site": "stackoverflow"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    question_links = []
    for item in data.get("items", []):
        question_links.append(item["link"])

    return question_links

# Function to update visit count in the session
def update_visit_count(session):
    # Initialize visit count if not present in session
    if 'visit_count' not in session:
        session['visit_count'] = 0
    
    # Increment visit count
    session['visit_count'] += 1

if __name__ == "__main__":
    main()
    