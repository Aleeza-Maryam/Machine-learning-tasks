import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

# -------------------------------
# Load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -------------------------------
# Text preprocessing
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)        
    text = re.sub(r"[^a-zA-Z]", " ", text)  
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# -------------------------------
# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #7a0534, #080808);
        font-family: "Segoe UI", sans-serif;
        color: white;
    }
    .title {
        text-align: center;
        color: white;
    }
    .muted {
        color: #dddddd;
        font-size: 0.95rem;
    }
    .conf-box {
        margin-top: 10px;
        padding: 12px 16px;
        border-radius: 10px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Load vectorizer
with open("sentiment_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load models
with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

models = {
    "Naive Bayes": nb_model,
    "Logistic Regression": lr_model,
    "SVM": svm_model
}

# -------------------------------
# Streamlit App
st.markdown("<h1 class='title'>Sentiment Analysis of Movie Reviews &#127909;</h1>", unsafe_allow_html=True)
st.markdown("<p class='muted'>Enter a movie review below and see if it's <b>Positive</b> or <b>Negative</b>.</p>", unsafe_allow_html=True)

# Dropdown to choose model
model_choice = st.selectbox("Choose Model:", list(models.keys()))
model = models[model_choice]

user_input = st.text_area("Paste your movie review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("&#9757; Please enter a review!")
    else:
        cleaned_text = clean_text(user_input)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]

        # Show prediction (using HTML entities instead of literal emojis)
        if prediction == 1:
            st.success("&#9989; Positive Review")
        else:
            st.error("&#10060; Negative Review")

        # Confidence (if available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)[0]
            confidence = float(np.max(proba) * 100)
            pos_conf = float(proba[1] * 100)
            neg_conf = float(proba[0] * 100)

            st.markdown(
                f"""
                <div class='conf-box'>
                    <div><b>&#128202; Confidence:</b> {confidence:.2f}%</div>
                    <div class='muted'>Positive: {pos_conf:.2f}% &nbsp;&nbsp;|&nbsp;&nbsp; Negative: {neg_conf:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("&#9888;&#65039; Confidence score is not available for the SVM model.")
