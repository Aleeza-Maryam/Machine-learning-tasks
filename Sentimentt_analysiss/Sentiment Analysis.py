import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
import pickle
# Load IMDb dataset (CSV file from Kaggle)
data = pd.read_csv("Sentimentt_analysiss/IMDB Dataset.csv")

print(data.head())
print(data['sentiment'].value_counts())


nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)   # remove HTML
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data["clean_review"] = data["review"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["clean_review"])

y = data["sentiment"].map({"positive": 1, "negative": 0})  # Encode labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# naive bayes

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("F1 Score:", f1_score(y_test, y_pred_nb))

# logistic regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

# svm
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))

print("Classification Report (SVM):")
print(classification_report(y_test, y_pred_svm))

def predict_review(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    pred = svm.predict(vector)[0]
    return "Positive +" if pred == 1 else "Negative -"

print(predict_review("The movie was fantastic! The acting was superb."))
print(predict_review("Worst movie ever. Waste of time."))


# Save trained model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(svm, f)   # or logistic regression

# Save vectorizer
with open("sentiment_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save each model
with open("nb_model.pkl", "wb") as f:
    pickle.dump(nb, f)

with open("lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)

with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)