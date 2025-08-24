import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- One-time NLTK downloads (safe to keep here) ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# --- Load trained artifacts ---
with open("all_models.pkl", "rb") as f:
    models = pickle.load(f)             # dict: {"Logistic Regression": model, "Naive Bayes": model, ...}

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)         # fitted TfidfVectorizer

# Label mapping YOU used during training (you said: fake=0, real=1)
LABEL_NAMES = {0: "FAKE", 1: "REAL"}

# --- Preprocessing must match training ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def debug_vectorization(text, vectorizer):
    """Debug function to see which words are being used from the input"""
    features = vectorizer.get_feature_names_out()
    vector = vectorizer.transform([text])
    nonzero_indices = vector.nonzero()[1]
    nonzero_features = [features[i] for i in nonzero_indices]
    return nonzero_features

# --- Streamlit UI ---
st.markdown("<h1>&#128240; Fake News Detection App</h1>", unsafe_allow_html=True)
st.write("Paste a news headline or (preferably) a longer snippet. The app will predict **REAL** or **FAKE** and show confidence.")

# Choose which classifier to use
model_name = st.selectbox("Choose a model", list(models.keys()), index=0)
model = models[model_name]

user_text = st.text_area("Paste news text here (longer is better):", height=180)

col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("Predict")

if predict_btn:
    if not user_text or not user_text.strip():
        st.warning("&#9757; Please enter some text.")
    else:
        cleaned = clean_text(user_text)
        st.write("Cleaned text:", cleaned)  # Debug output
        
        X = vectorizer.transform([cleaned])
        nnz = X.nnz
        
        # Debug: Show which words were recognized
        recognized_words = debug_vectorization(cleaned, vectorizer)
        st.write("Recognized words:", recognized_words)
        
        st.caption(f"Non-zero TF-IDF features for your text: **{nnz}**")
        if nnz == 0:
            st.info(
                "Your text has no words in the model's vocabulary (likely too short or off-domain). "
                "Try pasting a longer snippet from the article."
            )

        # Predict label (0 or 1) then map to human label
        y_pred = int(model.predict(X)[0])
        label = LABEL_NAMES.get(y_pred, str(y_pred))

        # Confidence if available
        conf_text = ""
        proba = None
        try:
            proba = model.predict_proba(X)[0]           # [p(fake), p(real)] given our mapping
            conf_real = float(proba[1])                 # probability of REAL
            conf_fake = float(proba[0])                 # probability of FAKE
            confidence = conf_real if y_pred == 1 else conf_fake
            conf_text = f" (confidence: {confidence*100:.1f}%)"
        except Exception:
            pass

        # Show result (HTML entity emojis)
        if y_pred == 1:
            st.success(f"&#9989; Prediction: **REAL**{conf_text}")
        else:
            st.error(f"&#10060; Prediction: **FAKE**{conf_text}")

        # Optional: show raw probabilities if available
        if proba is not None:
            st.write("Probabilities:")
            st.write({"FAKE (0)": round(conf_fake, 4), "REAL (1)": round(conf_real, 4)})

        # Tips to improve accuracy for manual tests
        st.markdown(
            """
**Tips:**  
- Paste 3–6 sentences from the article rather than a very short headline.  
- This model was trained on a political-news dataset; generic science/tech one-liners may be out-of-vocabulary.
"""
        )
