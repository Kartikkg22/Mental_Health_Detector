import sys
import os

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pickle
from src.preprocess import clean_text

# ✅ Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ✅ Load model + vectorizer
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# 🎨 UI Config
st.set_page_config(page_title="Mental Health Detector", page_icon="🧠")

# 🧠 Title
st.title("🧠 Mental Health Detection App")
st.write("Analyze text to detect potential mental health conditions")

# 📝 Input
user_input = st.text_area("Enter your text here:")

# 🚀 Button
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized).max()

        st.subheader("Prediction:")
        st.success(prediction)

        st.subheader("Confidence:")
        st.write(f"{round(probability * 100, 2)}%")