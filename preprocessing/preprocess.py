import pandas as pd
import re
import numpy as np
import nltk
nltk.download('popular')
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
from wordcloud import WordCloud


ques = pd.read_csv("Questions.csv",encoding="ISO-8859-1")
ques.head()
ques.info()

tags = pd.read_csv("Tags.csv",encoding='ISO-8859-1')
tags.head()
tags.info()

tags['Tag'] = tags['Tag'].astype(str)
grouped_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))
grouped_tags.reset_index()
grouped_tags.head()

ques.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
ques = ques.merge(grouped_tags, on='Id')
ques.head()

ques = ques[ques['Score']>5]
ques.info()

ques.drop(columns = ['Id','Score'],inplace=True)
ques.head()

del tags
del grouped_tags

ques['Tag'] = ques['Tag'].apply(lambda x: x.split())
all_tags = [item for sublist in ques['Tag'].values for item in sublist]
len(all_tags)

unique_tags = list(set(all_tags))
len(unique_tags)

ques.head()

counter = Counter(all_tags)
most_occur = counter.most_common(100)

tags = [i[0] for i in most_occur]
count = [i[1] for i in most_occur]
tags[:5],count[:5]

x = np.arange(len(most_occur))
fig, ax = plt.subplots(figsize=(20, 10))
plt.bar(x, height= count)
plt.xticks(x, tags, rotation=75)
plt.show()

def generate_wordcloud(text):
    wordcloud = WordCloud(width=3000,height = 1500).generate(text)
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

generate_wordcloud(" ".join(all_tags))

def most_common(x):
    tags_filtered = []
    for i in range(0, len(x)):
        if x[i] in tags:
            tags_filtered.append(x[i])
    return tags_filtered

ques['Tag'] = ques['Tag'].apply(lambda x: most_common(x))
ques['Tag'] = ques['Tag'].apply(lambda x: x if len(x)>0 else None)
ques.shape
ques.dropna(subset=['Tag'], inplace=True)
ques.shape
ques.head()

#nltk.download('popular')

token=ToktokTokenizer()
punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def strip_list_noempty(mylist):
    newlist = [item.strip() if hasattr(item, 'strip') else item for item in mylist]
    return [item for item in newlist if item != '']


def clean_punct(text):
    words = token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in tags:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))

    filtered_list = strip_list_noempty(punctuation_filtered)

    return ' '.join(map(str, filtered_list))


lemma = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def lemitizeWords(text):
    words = token.tokenize(text)
    listLemma = []
    for w in words:
        x = lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))


def stopWordsRemove(text):
    stop_words = set(stopwords.words("english"))

    words = token.tokenize(text)

    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered))


def combine(new_df):
    new_df['Body'] = new_df['Body'].apply(lambda x: clean_text(x))
    print(1)
    new_df['Body'] = new_df['Body'].apply(lambda x: clean_punct(x))
    print(2)
    new_df['Body'] = new_df['Body'].apply(lambda x: lemitizeWords(x))
    print(3)
    new_df['Body'] = new_df['Body'].apply(lambda x: stopWordsRemove(x))
    print(4)
    new_df['Title'] = new_df['Title'].apply(lambda x: str(x))
    print(5)
    new_df['Title'] = new_df['Title'].apply(lambda x: clean_text(x))
    print(6)
    new_df['Title'] = new_df['Title'].apply(lambda x: clean_punct(x))
    print(7)
    new_df['Title'] = new_df['Title'].apply(lambda x: lemitizeWords(x))
    print(8)
    new_df['Title'] = new_df['Title'].apply(lambda x: stopWordsRemove(x))
    return new_df


ques = combine(ques)
ques.head()

y = ques['Tag']
ques = ques.values
X = []

for i in tqdm(ques):
    X.append(i[0] + ' ' + i[1])

from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
Y = multilabel_binarizer.fit_transform(y)

len(X), Y.shape

X[0]


ques = pd.DataFrame(ques, columns=['Title', 'Body', 'Tag'])
ques.to_csv('final-data.csv', index=False)
