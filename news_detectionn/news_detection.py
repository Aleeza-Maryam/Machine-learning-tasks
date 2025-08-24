import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load CSVs
fake = pd.read_csv("news_detectionn/Fake.csv")
true = pd.read_csv("news_detectionn/True.csv")

# Add labels (Fake=0, True=1)
fake["label"] = 0
true["label"] = 1

# Combine into one dataset
data = pd.concat([fake, true], axis=0).reset_index(drop=True)

print("Original class distribution:")
print(data["label"].value_counts())

# Balance the dataset
fake_df = data[data["label"] == 0]
true_df = data[data["label"] == 1]

# Downsample the majority class (fake news)
fake_downsampled = resample(fake_df,
                           replace=False,
                           n_samples=len(true_df),
                           random_state=42)

# Combine back
balanced_data = pd.concat([fake_downsampled, true_df])

# Use balanced_data instead of data for the rest of the code
data = balanced_data

print("Balanced class distribution:")
print(data["label"].value_counts())

# Text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Check if text is not a string
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    # Remove numbers & special characters but keep basic punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Tokenize (split into words)
    words = text.split()
    # Remove stopwords + Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

# Apply cleaning
data["clean_text"] = data["text"].apply(clean_text)

print(data[["text","clean_text"]].head())

# Split data
X = data["clean_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# Vectorization with improved parameters
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=2,            # Ignore terms that appear in fewer than 2 documents
    max_df=0.8           # Ignore terms that appear in more than 80% of documents
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Shape of TF-IDF:", X_train_tfidf.shape)

# Model training
# Logistic Regression
lr = LogisticRegression(max_iter=200, class_weight='balanced')
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n {model_name} Results")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

# Evaluate all models
evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_nb, "Naive Bayes")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Save models and vectorizer
models = {
    "Logistic Regression": lr,
    "Naive Bayes": nb,
    "Random Forest": rf
}

with open("all_models.pkl", "wb") as f:
    pickle.dump(models, f)

# Save vectorizer too
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Models and vectorizer saved successfully!")