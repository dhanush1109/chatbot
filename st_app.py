import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load saved models
vectorizer = joblib.load('vectorizer.pkl')
X = joblib.load('tfidf_matrix.pkl')

# Load data from CSV file
data = pd.read_csv('Log_errors.csv')  # Adjust the file path as needed

# Define function to find most similar question
def get_answer(query, threshold=0.1):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(X, query_vec).flatten()
    max_similarity = similarities.max()
    if max_similarity < threshold:
        return "Insufficient information"
    else:
        most_similar_index = similarities.argmax()
        corresponding_answer = data.iloc[most_similar_index]["Troubleshoot"]
        return corresponding_answer

# Streamlit app
def main():
    st.title('Chatbot')

    user_query = st.text_input('Ask a question:')
    if user_query:
        if user_query.lower() == "exit":
            st.write("Goodbye!")
        else:
            answer = get_answer(user_query)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
