import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df = pd.read_csv("data/faq_dataset.csv")
df['cleaned_query'] = df['Query'].apply(preprocess)

X_train = df['cleaned_query']
y_train = df['Intent']

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(penalty='l2', solver='liblinear')
model.fit(X_train_vec, y_train)

# Save models
with open("models/intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)