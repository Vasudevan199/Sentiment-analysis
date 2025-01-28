import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# Load datasets
train_data = 'dataset path of the train data'
test_data = 'dataset path of the test data'

print(train_data.shape)
print(test_data.shape)

train_data.head()
test_data.head()

train_data.isnull().any()
test_data.isnull().any()

# Checking out negative comments from the train set
train_data[train_data['label'] == 0].head(10)

# Checking out positive comments from the train set
train_data[train_data['label'] == 1].head(10)

train_data['label'].value_counts().plot.bar(color='pink', figsize=(6, 4))

# Checking the distribution of tweets in the data
train_tweet_length = train_data['tweet'].str.len().plot.hist(color='pink', figsize=(6, 4))
test_tweet_length = test_data['tweet'].str.len().plot.hist(color='orange', figsize=(6, 4))

# Adding a column to represent the length of the tweet
train_data['tweet_length'] = train_data['tweet'].str.len()
test_data['tweet_length'] = test_data['tweet'].str.len()

train_data.head(10)
train_data.groupby('label').describe()

train_data.groupby('tweet_length').mean()['label'].plot.hist(color='black', figsize=(6, 4))
plt.title('Variation of Tweet Length')
plt.xlabel('Length')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
word_counts = vectorizer.fit_transform(train_data['tweet'])

word_sums = word_counts.sum(axis=0)

word_frequencies = [(word, word_sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
word_frequencies = sorted(word_frequencies, key=lambda x: x[1], reverse=True)

frequency_df = pd.DataFrame(word_frequencies, columns=['word', 'frequency'])

frequency_df.head(30).plot(x='word', y='frequency', kind='bar', figsize=(15, 7), color='blue')
plt.title("Most Frequently Occurring Words - Top 30")

from wordcloud import WordCloud

wordcloud = WordCloud(background_color='white', width=1000, height=1000).generate_from_frequencies(dict(word_frequencies))

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize=22)

neutral_words = ' '.join([text for text in train_data['tweet'][train_data['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=0, max_font_size=110).generate(neutral_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Neutral Words')
plt.show()

negative_words = ' '.join([text for text in train_data['tweet'][train_data['label'] == 1]])

wordcloud = WordCloud(background_color='cyan', width=800, height=500, random_state=0, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Negative Words')
plt.show()

# Collecting hashtags
def extract_hashtags(tweet_list):
    hashtags = []
    for tweet in tweet_list:
        hashtags.append(re.findall(r"#(\w+)", tweet))
    return hashtags

# Extracting hashtags from non-racist/sexist tweets
regular_hashtags = extract_hashtags(train_data['tweet'][train_data['label'] == 0])

# Extracting hashtags from racist/sexist tweets
negative_hashtags = extract_hashtags(train_data['tweet'][train_data['label'] == 1])

# Unnesting lists
regular_hashtags = sum(regular_hashtags, [])
negative_hashtags = sum(negative_hashtags, [])

regular_freq_dist = nltk.FreqDist(regular_hashtags)
regular_df = pd.DataFrame({'Hashtag': list(regular_freq_dist.keys()),
                           'Count': list(regular_freq_dist.values())})

# Selecting top 20 most frequent hashtags
regular_df = regular_df.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=regular_df, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.show()

negative_freq_dist = nltk.FreqDist(negative_hashtags)
negative_df = pd.DataFrame({'Hashtag': list(negative_freq_dist.keys()),
                            'Count': list(negative_freq_dist.values())})

# Selecting top 20 most frequent hashtags
negative_df = negative_df.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=negative_df, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.show()

# Tokenizing words in the training set
tokenized_tweets = train_data['tweet'].apply(lambda x: x.split())

# Importing gensim
import gensim

# Creating a word-to-vector model
w2v_model = gensim.models.Word2Vec(
    tokenized_tweets,
    size=200,
    window=5,
    min_count=2,
    sg=1,
    hs=0,
    negative=10,
    workers=2,
    seed=34
)

w2v_model.train(tokenized_tweets, total_examples=len(train_data['tweet']), epochs=20)

w2v_model.wv.most_similar(positive="dinner")
w2v_model.wv.most_similar(positive="cancer")
w2v_model.wv.most_similar(positive="apple")
w2v_model.wv.most_similar(negative="hate")

# Removing unwanted patterns from the data
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

train_cleaned = []

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

for tweet in train_data['tweet']:
    train_cleaned.append(clean_text(tweet))

test_cleaned = [clean_text(tweet) for tweet in test_data['tweet']]

# Creating bag of words
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=2500)
x_train_data = vectorizer.fit_transform(train_cleaned).toarray()
y_train_data = train_data['label']

x_test_data = vectorizer.transform(test_cleaned).toarray()

# Splitting the training data into train and validation sets
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train_data, y_train_data, test_size=0.25, random_state=42)

# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test_data = scaler.transform(x_test_data)

# Training and evaluating multiple models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

rf_pred = rf_model.predict(x_valid)
print("Random Forest Training Accuracy:", rf_model.score(x_train, y_train))
print("Random Forest Validation Accuracy:", rf_model.score(x_valid, y_valid))
print("Random Forest F1 Score:", f1_score(y_valid, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_valid, rf_pred))

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

lr_pred = lr_model.predict(x_valid)
print("Logistic Regression Training Accuracy:", lr_model.score(x_train, y_train))
print("Logistic Regression Validation Accuracy:", lr_model.score(x_valid, y_valid))
print("Logistic Regression F1 Score:", f1_score(y_valid, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_valid, lr_pred))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

dt_pred = dt_model.predict(x_valid)
print("Decision Tree Training Accuracy:", dt_model.score(x_train, y_train))
print("Decision Tree Validation Accuracy:", dt_model.score(x_valid, y_valid))
print("Decision Tree F1 Score:", f1_score(y_valid, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(y_valid, dt_pred))

# Support Vector Machine
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(x_train, y_train)

svc_pred = svc_model.predict(x_valid)
print("SVM Training Accuracy:", svc_model.score(x_train, y_train))
print("SVM Validation Accuracy:", svc_model.score(x_valid, y_valid))
print("SVM F1 Score:", f1_score(y_valid, svc_pred))
print("Confusion Matrix:\n", confusion_matrix(y_valid, svc_pred))

# XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)

xgb_pred = xgb_model.predict(x_valid)
print("XGBoost Training Accuracy:", xgb_model.score(x_train, y_train))
print("XGBoost Validation Accuracy:", xgb_model.score(x_valid, y_valid))
print("XGBoost F1 Score:", f1_score(y_valid, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_valid, xgb_pred))
