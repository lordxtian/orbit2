import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

from nlp.ner import NERModule
from fallback.qwen_fallback import QwenFallback
import config

@st.cache_resource
def load_models():
    with open("models/intent_model.pkl", "rb") as f:
        intent_model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    ner = NERModule()
    qwen = QwenFallback(config.DASHSCOPE_API_KEY)
    return intent_model, vectorizer, ner, qwen

intent_model, vectorizer, ner, qwen = load_models()

df = pd.read_csv("data/faq_dataset.csv")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

st.title("orBIT: Your Academic Assistant")
st.write("Ask questions about IT, CS, and EMC programs!")

user_input = st.text_input("Type your question here:")

if user_input:
    cleaned = preprocess(user_input)
    vec = vectorizer.transform([cleaned])
    intent = intent_model.predict(vec)[0]
    proba = intent_model.predict_proba(vec).max()

    entities = ner.extract_entities(user_input)

    if proba < 0.6:
        answer = qwen.get_response(user_input)
    else:
        answer = df[df["Intent"] == intent]["Answer"].iloc[0]

    st.markdown("### 🔍 Intent Detected:")
    st.write(intent)

    st.markdown("### 🧾 Entities Identified:")
    st.write(entities)

    st.markdown("### 💬 Response:")
    st.write(answer)