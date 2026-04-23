import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="Hypertension Predictor", layout="centered")

st.title("🫀 Hypertension Risk Predictor")
st.write("Enter patient details to estimate hypertension risk")

# -------------------------------
# USER INPUTS
# -------------------------------
age = st.slider("Age", 18, 100, 30)
salt_intake = st.slider("Salt Intake (1-10)", 1, 10, 5)
stress_score = st.slider("Stress Score (1-10)", 1, 10, 5)
sleep_duration = st.slider("Sleep Duration (hours)", 3, 12, 7)
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

bp_history = st.selectbox("BP History", ["No", "Yes"])
family_history = st.selectbox("Family History", ["No", "Yes"])
exercise_level = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])
smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])

medication = st.selectbox(
    "Medication",
    ["Beta Blocker", "Diuretic", "Other", "Unknown"]
)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict Risk"):

    # -------------------------------
    # ENCODING
    # -------------------------------
    bp_history_val = 1 if bp_history == "Yes" else 0
    family_history_val = 1 if family_history == "Yes" else 0
    smoking_status_val = 1 if smoking_status == "Yes" else 0

    exercise_map = {"Low": 0, "Moderate": 1, "High": 2}
    exercise_level_val = exercise_map[exercise_level]

    med_beta = 1 if medication == "Beta Blocker" else 0
    med_diuretic = 1 if medication == "Diuretic" else 0
    med_other = 1 if medication == "Other" else 0
    med_unknown = 1 if medication == "Unknown" else 0

    # -------------------------------
    # DATAFRAME
    # -------------------------------
    input_data = pd.DataFrame({
        'age': [age],
        'salt_intake': [salt_intake],
        'stress_score': [stress_score],
        'bp_history': [bp_history_val],
        'sleep_duration': [sleep_duration],
        'bmi': [bmi],
        'family_history': [family_history_val],
        'exercise_level': [exercise_level_val],
        'smoking_status': [smoking_status_val],
        'medication_Beta Blocker': [med_beta],
        'medication_Diuretic': [med_diuretic],
        'medication_Other': [med_other],
        'medication_Unknown': [med_unknown]
    })

   
    # SCALING
    # -------------------------------
    num_cols = ['age', 'salt_intake', 'stress_score', 'sleep_duration', 'bmi']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # -------------------------------
    # COLUMN ORDER
    # -------------------------------
    input_data = input_data[
        ['age', 'salt_intake', 'stress_score', 'bp_history', 'sleep_duration',
         'bmi', 'family_history', 'exercise_level', 'smoking_status',
         'medication_Beta Blocker', 'medication_Diuretic',
         'medication_Other', 'medication_Unknown']
    ]

    # -------------------------------
    # PREDICTION
    # -------------------------------
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # -------------------------------
    # OUTPUT
    # -------------------------------
    if prediction == 1:
        st.error("⚠️ High Risk of Hypertension")
    else:
        st.success("✅ Low Risk of Hypertension")

    st.progress(float(prob))
    st.write(f"Risk Score: {prob*100:.2f}%")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Model optimized for high recall (to minimize missed patients)")
st.warning("This prediction is for educational purposes only and not a medical diagnosis.")
