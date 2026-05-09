#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for streamlit
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #e74c3c, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #3498db;
    }
    .prediction-positive {
        background: #ffeaa7;
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-negative {
        background: #d5f5e3;
        border: 2px solid #2ecc71;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Models (cached so they load only once) ───────────────
@st.cache_resource
def load_models():
    heart_model    = joblib.load('models/heart_model.pkl')
    diab_model     = joblib.load('models/diabetes_model.pkl')
    heart_scaler   = joblib.load('models/heart_scaler.pkl')
    diab_scaler    = joblib.load('models/diabetes_scaler.pkl')

    with open('models/heart_features.json') as f:
        heart_features = json.load(f)
    with open('models/diabetes_features.json') as f:
        diab_features = json.load(f)
    with open('models/metadata.json') as f:
        metadata = json.load(f)

    return (heart_model, diab_model,
            heart_scaler, diab_scaler,
            heart_features, diab_features, metadata)

(heart_model, diab_model,
 heart_scaler, diab_scaler,
 heart_features, diab_features, metadata) = load_models()


# ── Preprocessing Functions ───────────────────────────────────
def preprocess_heart(inputs: dict) -> np.ndarray:
    df = pd.DataFrame([inputs])
    df['high_chol']  = (df['chol'] > 200).astype(int)
    df['high_bp']    = (df['trestbps'] > 130).astype(int)
    df['hr_reserve'] = df['thalach'] - df['trestbps']
    df = df.reindex(columns=heart_features, fill_value=0)
    arr    = np.array(df.values, dtype=np.float64)
    scaled = heart_scaler.transform(arr)
    return scaled.astype(np.float32)

def preprocess_diabetes(inputs: dict) -> np.ndarray:
    df = pd.DataFrame([inputs])
    df['glucose_insulin_ratio'] = df['Glucose'] / (df['Insulin'] + 1)
    df['age_risk']              = (df['Age'] > 45).astype(int)
    df['preg_age_risk']         = df['Pregnancies'] * df['Age']
    df = df.reindex(columns=diab_features, fill_value=0)
    arr    = np.array(df.values, dtype=np.float64)
    scaled = diab_scaler.transform(arr)
    return scaled.astype(np.float32)


# ── SHAP Explanation Plot ─────────────────────────────────────
def get_shap_plot(model, input_arr, feature_names, title):
    explainer   = shap.TreeExplainer(model)
    explanation = explainer(input_arr)

    if len(explanation.values.shape) == 3:
        shap_vals = explanation.values[0, :, 1]
        base_val  = explainer.expected_value[1]
    else:
        shap_vals = explanation.values[0]
        base_val  = explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[1]

    n = len(feature_names)
    shap_vals = shap_vals[:n]

    exp = shap.Explanation(
        values=shap_vals,
        base_values=float(base_val),
        data=input_arr[0][:n],
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(exp, show=False)
    plt.title(title, fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
#                        MAIN APP
# ════════════════════════════════════════════════════════════

# Header
st.markdown('<p class="main-header">🏥 Disease Prediction System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered Early Detection for Heart Disease & Diabetes</p>',
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
    st.title("Navigation")
    page = st.radio("Select Disease",
                    ["🏠 Home", "🫀 Heart Disease", "🩸 Diabetes",
                     "📊 Model Performance"])
    st.markdown("---")
    st.markdown("### About")
    st.info("This system uses Machine Learning to predict disease risk based on clinical parameters.")
    st.markdown("### ⚠️ Disclaimer")
    st.warning("This tool is for educational purposes only. Always consult a qualified doctor.")


# ════════════════════════════════════════════════════════════
#  HOME PAGE
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 🫀 Heart Disease Prediction")
        st.markdown(f"""
        - **Model:** {metadata['heart']['model_name']}
        - **Accuracy:** {metadata['heart']['metrics']['accuracy']}%
        - **ROC-AUC:** {metadata['heart']['metrics']['roc_auc']}%
        - **Features used:** {metadata['heart']['n_features']}
        """)
        st.success("Click '🫀 Heart Disease' in the sidebar to predict")

    with col2:
        st.markdown("## 🩸 Diabetes Prediction")
        st.markdown(f"""
        - **Model:** {metadata['diabetes']['model_name']}
        - **Accuracy:** {metadata['diabetes']['metrics']['accuracy']}%
        - **ROC-AUC:** {metadata['diabetes']['metrics']['roc_auc']}%
        - **Features used:** {metadata['diabetes']['n_features']}
        """)
        st.success("Click '🩸 Diabetes' in the sidebar to predict")

    st.markdown("---")
    st.markdown("## 🔬 How it works")
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**Step 1**\nEnter patient health parameters")
    col2.info("**Step 2**\nML model processes the input")
    col3.info("**Step 3**\nGet prediction with confidence")
    col4.info("**Step 4**\nSee SHAP explanation of why")


# ════════════════════════════════════════════════════════════
#  HEART DISEASE PAGE
# ════════════════════════════════════════════════════════════
elif page == "🫀 Heart Disease":
    st.markdown("## 🫀 Heart Disease Prediction")
    st.markdown("Enter the patient's clinical parameters below:")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.slider("Age (years)", 20, 80, 50)
        sex      = st.selectbox("Sex", [0, 1],
                                format_func=lambda x: "Female" if x == 0 else "Male")
        cp       = st.selectbox("Chest Pain Type",
                                [0, 1, 2, 3],
                                format_func=lambda x:
                                    ["Typical Angina", "Atypical Angina",
                                     "Non-anginal Pain", "Asymptomatic"][x])
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                               [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results",
                               [0, 1, 2],
                               format_func=lambda x:
                                   ["Normal", "ST-T Abnormality",
                                    "Left Ventricular Hypertrophy"][x])
        thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
        exang   = st.selectbox("Exercise Induced Angina",
                               [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        slope   = st.selectbox("Slope of Peak Exercise ST",
                               [0, 1, 2],
                               format_func=lambda x:
                                   ["Upsloping", "Flat", "Downsloping"][x])
        ca      = st.selectbox("Major Vessels Colored (0-3)",
                               [0, 1, 2, 3])
        thal    = st.selectbox("Thalassemia",
                               [0, 1, 2, 3],
                               format_func=lambda x:
                                   ["Normal", "Fixed Defect",
                                    "Reversible Defect", "Unknown"][x])

    st.markdown("---")

    if st.button("🔍 Predict Heart Disease", use_container_width=True):
        inputs = {
            'age': age, 'sex': sex, 'cp': cp,
            'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
            'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }

        with st.spinner("Analyzing patient data..."):
            arr  = preprocess_heart(inputs)
            pred = heart_model.predict(arr)[0]
            prob = heart_model.predict_proba(arr)[0][1]

        # Result
        st.markdown("### 📋 Prediction Result")
        col1, col2, col3 = st.columns(3)

        with col1:
            if pred == 1:
                st.error(f"⚠️ **HIGH RISK**\nHeart Disease Detected")
            else:
                st.success(f"✅ **LOW RISK**\nNo Heart Disease Detected")

        with col2:
            st.metric("Disease Probability", f"{prob*100:.1f}%")

        with col3:
            st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")

        # Gauge bar
        st.markdown("### 🎯 Risk Level")
        st.progress(float(prob))
        if prob < 0.3:
            st.success("Low Risk Zone")
        elif prob < 0.6:
            st.warning("Moderate Risk Zone")
        else:
            st.error("High Risk Zone")

        # SHAP
        st.markdown("### 🔍 Why did the model predict this?")
        st.markdown("*SHAP values show which features pushed the prediction up (red) or down (blue)*")
        with st.spinner("Generating explanation..."):
            fig = get_shap_plot(
                heart_model, arr,
                heart_features,
                "Heart Disease — Feature Contributions"
            )
            st.pyplot(fig)
            plt.close()


# ════════════════════════════════════════════════════════════
#  DIABETES PAGE
# ════════════════════════════════════════════════════════════
elif page == "🩸 Diabetes":
    st.markdown("## 🩸 Diabetes Prediction")
    st.markdown("Enter the patient's clinical parameters below:")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
        glucose     = st.slider("Glucose Level (mg/dl)", 0, 200, 120)
        bp          = st.slider("Blood Pressure (mmHg)", 0, 122, 70)
        skin        = st.slider("Skin Thickness (mm)", 0, 99, 20)

    with col2:
        insulin     = st.slider("Insulin Level (IU/ml)", 0, 846, 80)
        bmi         = st.slider("BMI", 0.0, 67.0, 25.0, 0.1)
        dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
        age         = st.slider("Age (years)", 21, 81, 35)

    st.markdown("---")

    if st.button("🔍 Predict Diabetes", use_container_width=True):
        inputs = {
            'Pregnancies'             : pregnancies,
            'Glucose'                 : glucose,
            'BloodPressure'           : bp,
            'SkinThickness'           : skin,
            'Insulin'                 : insulin,
            'BMI'                     : bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age'                     : age
        }

        with st.spinner("Analyzing patient data..."):
            arr  = preprocess_diabetes(inputs)
            pred = diab_model.predict(arr)[0]
            prob = diab_model.predict_proba(arr)[0][1]

        # Result
        st.markdown("### 📋 Prediction Result")
        col1, col2, col3 = st.columns(3)

        with col1:
            if pred == 1:
                st.error("⚠️ **HIGH RISK**\nDiabetes Detected")
            else:
                st.success("✅ **LOW RISK**\nNo Diabetes Detected")

        with col2:
            st.metric("Disease Probability", f"{prob*100:.1f}%")

        with col3:
            st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")

        # Gauge bar
        st.markdown("### 🎯 Risk Level")
        st.progress(float(prob))
        if prob < 0.3:
            st.success("Low Risk Zone")
        elif prob < 0.6:
            st.warning("Moderate Risk Zone")
        else:
            st.error("High Risk Zone")

        # SHAP
        st.markdown("### 🔍 Why did the model predict this?")
        st.markdown("*SHAP values show which features pushed the prediction up (red) or down (blue)*")
        with st.spinner("Generating explanation..."):
            fig = get_shap_plot(
                diab_model, arr,
                diab_features,
                "Diabetes — Feature Contributions"
            )
            st.pyplot(fig)
            plt.close()


# ════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE PAGE
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🫀 Heart Disease Model")
        st.markdown(f"**Algorithm:** {metadata['heart']['model_name']}")
        m = metadata['heart']['metrics']
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",  f"{m['accuracy']}%")
        c2.metric("F1 Score",  f"{m['f1_score']}%")
        c3.metric("ROC-AUC",   f"{m['roc_auc']}%")

    with col2:
        st.markdown("### 🩸 Diabetes Model")
        st.markdown(f"**Algorithm:** {metadata['diabetes']['model_name']}")
        m = metadata['diabetes']['metrics']
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",  f"{m['accuracy']}%")
        c2.metric("F1 Score",  f"{m['f1_score']}%")
        c3.metric("ROC-AUC",   f"{m['roc_auc']}%")

    st.markdown("---")
    st.markdown("### 🔬 About the Models")
    st.markdown("""
    | Feature | Heart Disease | Diabetes |
    |---|---|---|
    | Algorithm | Random Forest | XGBoost |
    | Train Size | 80% of dataset | 80% of dataset |
    | Test Size | 20% of dataset | 20% of dataset |
    | Scaling | StandardScaler | StandardScaler |
    | Class Handling | class_weight='balanced' | scale_pos_weight |
    | Explainability | SHAP TreeExplainer | SHAP TreeExplainer |
    """)


# In[ ]:




