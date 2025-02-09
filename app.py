import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)
    df["terms"] = (df["medical_term"] + " " + df["symptoms"]).apply(preprocess_text)
    return df

def compute_tfidf_vectors(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["terms"])
    return vectorizer, tfidf_matrix

def find_best_match(user_input, vectorizer, tfidf_matrix, df):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    best_match_idx = np.argmax(similarities)
    return df.iloc[best_match_idx], similarities[best_match_idx]

st.title("Medical Department Classifier")

csv_path = "data/large_data.csv"
df = load_data(csv_path)
vectorizer, tfidf_matrix = compute_tfidf_vectors(df)

user_input = st.text_input("Enter your medical term")
if user_input:
    best_match_row, score = find_best_match(user_input, vectorizer, tfidf_matrix, df)
    st.write(f"### Predicted Department: {best_match_row['departments']}")
    st.write(f"### Symptoms: {best_match_row['symptoms']}")
    st.write(f"### Related Conditions: {best_match_row['related_conditions']}")
