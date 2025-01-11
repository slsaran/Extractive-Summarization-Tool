# Text Summarization Tool

## Description
This tool uses logistic regression to rank sentences based on importance and generate concise summaries. It supports multiple levels of summarization (short, medium) and provides strong performance as measured by precision, recall, F1 score, and accuracy.

## Features
- Extractive summarization of text.
- Logistic regression-based model with TF-IDF features.
- Multiple levels of summarization (short, medium).

## Installation
 Clone the repository:
   ```bash
   git clone <repository-url>
   ```
 Navigate to the project folder
  ```bash
   cd <repository-folder>
   ```
 Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  Run the Streamlit app
  ```bash
  run main.py
  run python evaluation.py
  ```

## Evaluation Metrics
The logistic regression model has been evaluated on a custom labeled dataset, achieving excellent performance:

Precision: 0.9808
Recall: 0.9808
F1 Score: 0.9808
Accuracy: 0.9808

## Model Training
The summarization model was trained on a custom dataset of labeled sentences. The training process involved:

Feature Extraction: Sentences were vectorized using a TF-IDF vectorizer.
Logistic Regression: A logistic regression classifier was trained to rank sentences based on their importance in summaries.
Model Evaluation: The model's performance was validated using standard metrics such as precision, recall, F1 score, and accuracy.


## Files in Repository
requirements.txt: List of dependencies.
summarization_model.pkl: Trained logistic regression model.
tfidf_vectorizer.pkl: Pre-trained TF-IDF vectorizer.
evaluation.py: Script to evaluate model performance.
app.py: Streamlit-based GUI for summarization.

## Future Work
Transformer-Based Models: Explore transformer architectures (e.g., BERT, GPT) to improve summarization performance.
Enhanced Long Document Support: Improve summarization for lengthy documents by handling complex structures and semantics.

