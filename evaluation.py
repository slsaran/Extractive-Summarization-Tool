import joblib
import nltk
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load Model and Vectorizer
model = joblib.load('summarization_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing Function
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return ' '.join(tokens)

# Evaluation Function
def evaluate():
    # Sample data for training (can be extended with a larger dataset)
    data = [
        ("The quick brown fox jumps over the lazy dog.", 0),
        ("This is a significant sentence worth noting.", 1),
        ("Completely irrelevant information for testing.", 0),
        ("An essential detail in the context of our study.", 1),
        ("This is another trivial statement.", 0),
        ("Final important message for our dataset.", 1),
        ("Random noise added for distraction.", 0),
        ("Key data point that needs to be highlighted.", 1),
        ("Unnecessary fluff without much value.", 0),
        ("A critical conclusion to support our analysis.", 1),
        ("A random sentence with no real meaning.", 0),
        ("This piece of data is essential for decision making.", 1),
        ("Filler text that doesn't matter.", 0),
        ("This sentence contains valuable insights.", 1),
        ("A generic sentence with no real importance.", 0),
        ("This piece of information is quite crucial.", 1),
        ("Noise in the dataset that can be ignored.", 0),
        ("Important observation related to the research.", 1),
        ("Just some more filler data.", 0),
        ("Final crucial sentence for this section.", 1),
        ("This doesn't add much value.", 0),
        ("A critical finding from the experiment.", 1),
        ("A completely irrelevant piece of information.", 0),
        ("This sentence is very informative.", 1),
        ("Just a random sentence.", 0),
        ("An insightful conclusion from the report.", 1),
        ("Another filler sentence that can be ignored.", 0),
        ("This point is extremely important.", 1),
        ("A completely unnecessary distraction.", 0),
        ("A vital observation about the trends.", 1),
        ("Random text to pad out the dataset.", 0),
        ("This is a key takeaway from the study.", 1),
        ("This doesn't help with the analysis.", 0),
        ("An essential discovery in the research.", 1),
        ("A sentence that holds no significance.", 0),
        ("Critical insight that should be noted.", 1),
        ("Just another random statement.", 0),
        ("A crucial finding that backs our theory.", 1),
        ("Unnecessary information for testing.", 0),
        ("This sentence highlights an important detail.", 1),
        ("More random sentences with no value.", 0),
        ("A key conclusion drawn from the dataset.", 1),
        ("Filler text not needed for conclusions.", 0),
        ("This is an important insight.", 1),
        ("Random sentence without meaning.", 0),
        ("A vital piece of information.", 1),
        ("Another random sentence.", 0),
        ("An essential takeaway from the results.", 1),
        ("Unimportant detail that can be skipped.", 0),
        ("This is an important finding.", 1),
        ("Just filler text for the dataset.", 0),
        ("A key observation in the analysis.", 1),
        ("This doesn't contribute to the results.", 0),
        ("An important note for future reference.", 1),
        ("Completely irrelevant sentence.", 0),
        ("A significant point to remember.", 1),
        ("Random sentence for no reason.", 0),
        ("A crucial detail in the conclusion.", 1),
        ("This is just noise.", 0),
        ("An essential observation from the data.", 1),
        ("More irrelevant content.", 0),
        ("A final important finding.", 1),
        ("Filler text without purpose.", 0),
        ("This sentence is quite important.", 1),
        ("A sentence that doesn't matter.", 0),
        ("An important conclusion drawn from data.", 1),
        ("Random text for testing purposes.", 0),
        ("A vital point highlighted in the report.", 1),
        ("More random filler sentences.", 0),
        ("This is a crucial observation.", 1),
        ("This sentence adds no value.", 0),
        ("An insightful comment based on research.", 1),
        ("Random data added for no reason.", 0),
        ("A significant result from the analysis.", 1),
        ("Unimportant sentence for padding.", 0),
        ("An important observation related to trends.", 1),
        ("A sentence without any impact.", 0),
        ("This is an insightful conclusion.", 1),
        ("Just another random sentence.", 0),
        ("A key takeaway for further investigation.", 1),
        ("Filler text with no importance.", 0),
        ("This sentence is crucial to understanding.", 1),
        ("Random words with no meaning.", 0),
        ("A critical observation in the dataset.", 1),
        ("Just another random phrase.", 0),
        ("This is an important discovery.", 1),
        ("More filler data.", 0),
        ("A vital insight into the research.", 1),
        ("Another irrelevant sentence.", 0),
        ("A key observation in the findings.", 1),
        ("Random filler text.", 0),
        ("This is a crucial piece of information.", 1),
        ("Unimportant sentence without context.", 0),
        ("This is an important conclusion.", 1),
        ("Just random filler text.", 0),
        ("A significant point to note from the analysis.", 1),
        ("The stock market saw a significant downturn today due to inflation fears.", 1),
        ("The weather in California has been unusually mild this season.", 0),
        ("Quantum computing could revolutionize encryption methods.", 1),
        ("The book explores themes of love and betrayal in 18th-century Europe.", 0),
        ("This breakthrough in cancer research could save millions of lives.", 1),
        ("The team is preparing for the upcoming championship next month.", 0),
        ("The study provides critical insights into renewable energy adoption.", 1),
        ("A new restaurant opened downtown, offering Italian cuisine.", 0),
    ]
    df = pd.DataFrame(data, columns=['sentence', 'label'])
    df['cleaned'] = df['sentence'].apply(preprocess)
    X = tfidf.transform(df['cleaned']).toarray()
    y = df['label']
    y_pred = model.predict(X)

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    evaluate()
