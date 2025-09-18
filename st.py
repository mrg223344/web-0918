# app.py
# Peptic Ulcer Bleeding Rebleeding Risk Prediction Web App

import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier  # Correct import for classification

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Rebleeding Risk Prediction for Peptic Ulcer Bleeding",
    page_icon="🩸",
    layout="centered"
)

# -----------------------------
# Load Model (Cached to avoid reloading)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = CatBoostClassifier()
        model.load_model("catboost_model.cbm")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed. Please check if 'catboost_model.cbm' exists: {e}")
        st.stop()

model = load_model()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_rebleeding(method, location, descending, creatinine, bun, pt, aptt, rockall, aims65):
    try:
        data = pd.DataFrame({
            'Method': [int(method)],
            'Location': [int(location)],
            'Descending': [int(descending)],
            'Creatinine': [float(creatinine)],
            'BUN': [float(bun)],
            'PT': [float(pt)],
            'APTT': [float(aptt)],
            'Rockall': [int(rockall)],
            'AIMS65': [int(aims65)]
        })

        # Set categorical columns (must match training)
        data['Method'] = data['Method'].astype('category')
        data['Location'] = data['Location'].astype('category')
        data['Descending'] = data['Descending'].astype('category')

        # Predict probability of rebleeding (class 1: DN group)
        prob = model.predict_proba(data)[0][1]
        return max(0.0, min(1.0, prob))  # Clamp to [0,1]

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return 0.5  # Fallback probability

# -----------------------------
# User Interface (Fully in English)
# -----------------------------
st.title("🩸 Rebleeding Risk Prediction for Peptic Ulcer Bleeding")

st.markdown("""
> A machine learning model based on CatBoost to predict the risk of **rebleeding (DN group)** in patients with peptic ulcer bleeding.  
> Designed for post-endoscopic hemostasis risk stratification and clinical decision support.
""")

st.markdown("### 📝 Please enter patient clinical information:")

with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Treatment Method",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "1 - Clip",
                2: "2 - Electrocoagulation",
                3: "3 - Spray",
                4: "4 - Injection",
                5: "5 - Combined Therapy"
            }[x]
        )

        location = st.selectbox(
            "Lesion Location",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            format_func=lambda x: {
                1: "1 - Cardia",
                2: "2 - Gastric Body",
                3: "3 - Gastric Fundus",
                4: "4 - Gastric Angle",
                5: "5 - Gastric Antrum",
                6: "6 - Pylorus",
                7: "7 - Duodenal Bulb",
                8: "8 - Descending Duodenum",
                9: "9 - Anastomotic Stoma"
            }[x]
        )

        descending = st.radio(
            "Is lesion in descending duodenum?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        creatinine = st.number_input(
            "Creatinine (μmol/L)",
            min_value=0.0,
            max_value=2000.0,
            value=80.0,
            step=0.1
        )

    with col2:
        bun = st.number_input(
            "Blood Urea Nitrogen (BUN, mmol/L)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )
        pt = st.number_input(
            "Prothrombin Time (PT, seconds)",
            min_value=0.0,
            max_value=60.0,
            value=13.0,
            step=0.1
        )
        aptt = st.number_input(
            "Activated Partial Thromboplastin Time (APTT, seconds)",
            min_value=0.0,
            max_value=200.0,
            value=30.0,
            step=0.1
        )
        rockall = st.number_input(
            "Rockall Score",
            min_value=0,
            max_value=12,
            value=9,
            step=1
        )
        aims65 = st.number_input(
            "AIMS65 Score",
            min_value=0,
            max_value=5,
            value=3,
            step=1
        )

    submitted = st.form_submit_button("📊 Calculate Rebleeding Risk", type="primary")

# -----------------------------
# Display Prediction Result
# -----------------------------
if submitted:
    with st.spinner("Calculating risk..."):
        prob = predict_rebleeding(method, location, descending, creatinine, bun, pt, aptt, rockall, aims65)
        prob_percent = round(prob * 100, 1)

    # Risk level classification
    if prob < 0.4:
        risk_level = "Low Risk"
        color = "green"
        icon = "✅"
    elif prob < 0.7:
        risk_level = "Moderate Risk"
        color = "orange"
        icon = "⚠️"
    else:
        risk_level = "High Risk"
        color = "red"
        icon = "🚨"

    st.markdown("### 🔍 Prediction Result")
    st.markdown(f"{icon} **Risk Level:** <span style='color:{color}; font-weight:bold'>{risk_level}</span>", unsafe_allow_html=True)
    st.markdown(f"🩸 **Rebleeding Probability:** {prob_percent}%")
    st.progress(prob)

    st.markdown("---")
    st.markdown("#### 📌 Clinical Recommendations")
    if risk_level == "Low Risk":
        st.success("Low short-term rebleeding risk. Routine monitoring may be sufficient.")
    elif risk_level == "Moderate Risk":
        st.warning("Consider close observation and evaluate need for intensified therapy.")
    else:
        st.error("High-risk patient! Consider blood transfusion, ICU admission, or early repeat endoscopy.")

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("---")
st.markdown("""
**📌 Note:**
- This model was trained on retrospective data to predict membership in the DN (high-risk) group.
- For research and clinical decision support only. **Not a substitute for professional medical judgment.**
- Model file: `catboost_model.cbm`
""")