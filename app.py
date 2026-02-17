import streamlit as st
import pandas as pd
import io
import gspread
from google.oauth2.service_account import Credentials
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Diabetes Risk Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI-Based Diabetes Risk Detection")
st.markdown("Enter patient health details to assess diabetes risk.")
st.markdown("---")

# =====================================================
# TRAIN MODEL (Cloud Safe)
# =====================================================

@st.cache_resource
def train_model():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    df = pd.get_dummies(df, columns=["gender", "smoking_history"], drop_first=True)

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_estimators=200
    )

    model.fit(X_train, y_train)
    return model, X.columns

model, feature_columns = train_model()

# =====================================================
# GOOGLE SHEETS SAVE FUNCTION
# =====================================================

def save_to_google_sheet(data):
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ],
        )

        client = gspread.authorize(creds)
        sheet = client.open("user_data").sheet1  # <-- CHANGE if needed

        sheet.append_row([
            data["gender"],
            data["age"],
            data["hypertension"],
            data["heart_disease"],
            data["smoking_history"],
            data["bmi"],
            data["HbA1c_level"],
            data["blood_glucose_level"],
            data["prediction"],
            data["probability"]
        ])

        return True

    except Exception as e:
        st.error("❌ Failed to save data to Google Sheet")
        st.write(e)
        return False

# =====================================================
# PDF GENERATOR
# =====================================================

def generate_pdf(age, bmi, hba1c, glucose, probability):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Diabetes Risk Report", styles['Title']))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Age: {age}", styles['Normal']))
    elements.append(Paragraph(f"BMI: {bmi:.2f}", styles['Normal']))
    elements.append(Paragraph(f"HbA1c Level: {hba1c}", styles['Normal']))
    elements.append(Paragraph(f"Blood Glucose Level: {glucose}", styles['Normal']))
    elements.append(Paragraph(f"Risk Probability: {probability:.2%}", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# INPUT SECTION
# =====================================================

st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 30)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.subheader("❤️ Medical Conditions")

hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])

st.subheader("⚖ Body Information")

col1, col2 = st.columns(2)
with col1:
    height_cm = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
with col2:
    weight_kg = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)

bmi = weight_kg / ((height_cm / 100) ** 2)
st.markdown(f"### 📊 Calculated BMI: **{bmi:.2f}**")

if bmi < 18.5:
    st.info("Underweight")
elif bmi < 25:
    st.success("Normal weight")
elif bmi < 30:
    st.warning("Overweight")
else:
    st.error("Obese")

st.subheader("🚬 Lifestyle")

smoking = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "No Info"]
)

st.subheader("🧪 Blood Report")

with st.expander("ℹ What is HbA1c?"):
    st.write("""
    HbA1c measures average blood sugar over 2–3 months.
    Normal: < 5.7%
    Prediabetes: 5.7–6.4%
    Diabetes: ≥ 6.5%
    """)

hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.5)

with st.expander("ℹ What is Blood Glucose Level?"):
    st.write("""
    Blood Glucose Level shows current blood sugar.
    Normal fasting: 70–99 mg/dL
    Prediabetes: 100–125 mg/dL
    Diabetes: ≥ 126 mg/dL
    """)

glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 400, 120)

st.markdown("---")

# =====================================================
# PREDICTION
# =====================================================

if st.button("🔍 Check Diabetes Risk"):

    input_data = pd.DataFrame({
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose],
        "gender_Male": [1 if gender == "Male" else 0],
        "smoking_history_current": [1 if smoking == "current" else 0],
        "smoking_history_former": [1 if smoking == "former" else 0],
        "smoking_history_never": [1 if smoking == "never" else 0]
    })

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.metric("Risk Probability", f"{probability:.2%}")
    st.progress(int(probability * 100))

    # Save Data
    saved = save_to_google_sheet({
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "prediction": prediction,
        "probability": probability
    })

    if saved:
        st.success("Data saved to Google Sheet ✅")

    # PDF
    pdf = generate_pdf(age, bmi, hba1c, glucose, probability)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf,
        file_name="diabetes_report.pdf",
        mime="application/pdf"
    )

    st.caption("⚠️ This tool is for educational purposes only.")
