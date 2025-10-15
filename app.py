import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load model
model = joblib.load("models/breast_cancer_model.pkl")

st.title("🔬 Breast Cancer Detection App")
st.write("Enter the following tumor features to predict whether it's **malignant** or **benign**:")

# Get feature names
features = load_breast_cancer().feature_names

# User input for all features
user_input = []
for feature in features:
    value = st.number_input(f"{feature}", format="%.5f")
    user_input.append(value)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=features)
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df).max()

    result = "🔴 Malignant" if prediction == 0 else "🟢 Benign"
    st.subheader(f"Prediction: {result}")
    st.caption(f"Confidence: {confidence:.2%}")

# Feature importance visualization

st.subheader("📊 Feature Importance (Top 10)")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
top_features = [features[i] for i in indices]
top_importances = importances[indices]

fig, ax = plt.subplots()
ax.barh(top_features[::-1], top_importances[::-1])
ax.set_xlabel("Importance Score")
ax.set_title("Top 10 Feature Importances")
st.pyplot(fig)

st.markdown("---")
st.subheader("📁<--ADD from Docs")
uploaded_file = st.file_uploader("Upload a Doc with the same 30 features mentioned[CSV]", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        preds = model.predict(df)
        probs = model.predict_proba(df).max(axis=1)

        df["Prediction"] = ["Malignant" if p == 0 else "Benign" for p in preds]
        df["Confidence"] = [f"{c:.2%}" for c in probs]
        
        st.write("### ✅ Predictions:")
        st.dataframe(df)

        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", csv_download, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}\n\nEnsure the uploaded file has **exactly the same columns** as required.")

import seaborn as sns

if uploaded_file:

    # Plot prediction counts
    st.subheader("📈 Prediction Summary")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Prediction", data=df, palette=["red", "green"], ax=ax2)
    ax2.set_title("Benign vs Malignant Count")
    st.pyplot(fig2)
st.info("⚠️ Please upload a CSV with **exactly the same 30 columns** as the training data. You can download a template below.")

with open("sample_input_template.csv", "rb") as f:
    st.download_button("📥 Download Input CSV Template", f, "sample_input_template.csv", "text/csv")
