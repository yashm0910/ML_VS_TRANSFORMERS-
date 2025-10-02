import streamlit as st
import joblib
import shap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

dataset = load_dataset("ag_news")
label_names = dataset["train"].features["label"].names

ml_models = {
    "Logistic Regression": joblib.load("ml_models/logistic_regression.pkl"),
    "Naive Bayes": joblib.load("ml_models/multinomial_naive_bayes.pkl"),
    "SVM": joblib.load("ml_models/support_vector_machine.pkl"),
    "Random Forest": joblib.load("ml_models/random_forest.pkl"),
    "Gradient Boosting": joblib.load("ml_models/gradient_boosting.pkl"),
    "KNN": joblib.load("ml_models/k-nearest_neighbors.pkl"),
}

ml_vectorizer = joblib.load("ml_models/tfidf_vectorizer.pkl") 

transformer_path = "transformer_model1"
tokenizer = AutoTokenizer.from_pretrained(transformer_path)
transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_path)

st.set_page_config(page_title="News Classifier", layout="wide")

st.title(" Algorithm Warfare: Transformers vs Traditional ML Showdown ")
st.write("Choose between ML models or Transformer to classify a news article.")

choice = st.radio("Select Model Type:", ["Transformer", "Machine Learning"])

ml_choice = None
if choice == "Machine Learning":
    ml_choice = st.selectbox("Pick an ML model:", list(ml_models.keys()))


index = st.slider("Select an article index", 0, len(dataset["test"]) - 1, 0)
text = dataset["test"][index]["text"]
true_label = label_names[dataset["test"][index]["label"]]

st.markdown(f"**📰 Article:** {text}")
st.markdown(f"**✅ True Label:** {true_label}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 Transformer Prediction")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_label = label_names[np.argmax(probs)]
    st.write(f"**Predicted:** {pred_label}")
    st.bar_chart(probs)

with col2:
    st.subheader("⚡ ML Model Prediction")
    if ml_choice:
        model = ml_models[ml_choice]
        vec = ml_vectorizer.transform([text])
        probs = model.predict_proba(vec)[0]
        pred_label = label_names[np.argmax(probs)]
        st.write(f"**Predicted:** {pred_label}")
        st.bar_chart(probs)
    else:
        st.info("Select an ML model to see predictions.")

st.subheader("🔍 SHAP Explanation")
if choice == "Machine Learning" and ml_choice:
    # SHAP for ML
    explainer = shap.Explainer(ml_models[ml_choice], ml_vectorizer.transform)
    shap_values = explainer([text])
    st.pyplot(shap.plots.text(shap_values[0], show=False))
else:
    # SHAP for Transformer
    explainer = shap.Explainer(transformer_model, tokenizer)
    shap_values = explainer([text])
    st.pyplot(shap.plots.text(shap_values[0], show=False))