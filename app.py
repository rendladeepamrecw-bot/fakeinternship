import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="centered")

st.markdown("<h1 style='color:#2196F3; text-align:center;'>🤖 AI Internship Detector</h1>", unsafe_allow_html=True)

# Dummy dataset
data = pd.DataFrame({
    "company_score": [1,0,1,1,0,1],
    "email_valid": [1,0,1,1,0,1],
    "link_valid": [1,0,1,0,0,1],
    "stipend": [1,0,1,0,0,1],
    "label": [1,0,1,0,0,1]  # 1=Real, 0=Fake
})

X = data.drop("label", axis=1)
y = data["label"]

model = LogisticRegression()
model.fit(X, y)

# Inputs
company = st.text_input("🏢 Company Name")
email = st.text_input("📧 Email")
link = st.text_input("🔗 Link")
stipend = st.radio("💰 Stipend?", ["Yes", "No"])
image = st.file_uploader("🖼 Upload Image")

if st.button("🔍 Predict"):
    input_data = np.array([[
        1 if company else 0,
        1 if "@" in email else 0,
        1 if "https" in link else 0,
        1 if stipend == "Yes" else 0
    ]])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1 and prob > 0.7:
        result, emoji, color = "Real", "✅", "green"
    elif prob > 0.4:
        result, emoji, color = "Suspicious", "⚠️", "orange"
    else:
        result, emoji, color = "Fake", "❌", "red"

    st.markdown(f"<h2 style='color:{color}; text-align:center;'>{emoji} {result}</h2>", unsafe_allow_html=True)

    st.write("Confidence:", round(prob*100,2), "%")
