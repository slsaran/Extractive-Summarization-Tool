import streamlit as st
import joblib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

# Load the pre-trained model and tfidf vectorizer
with open('q1/summarization_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

with open('q1/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = joblib.load(vectorizer_file)

# Define your summarize function
def summarize(text, model, tfidf, level='Medium'):
    """
    Summarize the input text using the pre-trained model and tfidf vectorizer.
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Create TF-IDF features
    tfidf_matrix = tfidf.transform([text])
    
    # Use the model to predict the summary
    summary = model.predict(tfidf_matrix)
    
    # Assuming 'summary' contains an index of sentences to extract, adjust accordingly
    if level == 'Medium':
        num_sentences = 5  # Adjust for "Medium" level
    else:
        num_sentences = 2  # Adjust for other levels
    
    # Extract the top 'num_sentences' sentences as the summary
    summary_sentences = sentences[:num_sentences]
    
    # Return the summary as a concatenated string of sentences
    return " ".join(summary_sentences)

# Streamlit App
st.title("Text Summarization App")
st.write("Enter your text below to get a summarized version.")

# Input text from the user
input_text = st.text_area("Enter Text Here", height=200)

# Select summarization level
level = st.selectbox("Select Summarization Level", options=["Medium", "Short"], index=0)

# Summarize button
if st.button("Summarize"):
    if input_text.strip():  # Check if text is entered
        summary = summarize(input_text, model, tfidf, level)
        st.subheader("Summarized Text:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
