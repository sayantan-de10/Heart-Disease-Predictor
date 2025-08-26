import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("heart_rf_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

# Load template
template = pd.read_csv("Heart_user_template.csv")

st.title("🫀 Heart Disease Prediction App")
st.write("Fill in the form below to predict the risk of heart disease.")

# Predefined categorical mappings (based on UCI dataset)
sex_map = {"Male": 1, "Female": 0}
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3,
}
fbs_map = {"True (Fasting Blood Sugar > 120 mg/dl)": 1, "False": 0}
restecg_map = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left Ventricular Hypertrophy": 2,
}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}

# Form for input
with st.form("heart_form"):
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", list(sex_map.keys()))
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_map.keys()))
    restecg = st.selectbox("Resting ECG Results", list(restecg_map.keys()))
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", list(exang_map.keys()))
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", list(slope_map.keys()))
    ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", list(thal_map.keys()))

    submit = st.form_submit_button("Predict")

if submit:
    # Prepare input row
    user_input = {
        "age": age,
        "sex": sex_map[sex],
        "cp": cp_map[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_map[fbs],
        "restecg": restecg_map[restecg],
        "thalach": thalach,
        "exang": exang_map[exang],
        "oldpeak": oldpeak,
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal_map[thal],
    }

    user_df = pd.DataFrame([user_input])

    # Align with training template
    user_df_encoded = pd.get_dummies(user_df)
    user_df_encoded = user_df_encoded.reindex(columns=template.columns, fill_value=0)

    # Scale
    user_scaled = scaler.transform(user_df_encoded)

    # Predict
    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts: **High Risk of Heart Disease**")
    else:
        st.success("✅ The model predicts: **Low Risk of Heart Disease**")
