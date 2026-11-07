
import pickle
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b132b, #1c2541);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

div.stButton > button {
    background-color: #00b4d8;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 25px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #0096c7;
    color: white;
}

[data-testid="stMetricValue"] {
    color: #00b4d8;
}
</style>
""", unsafe_allow_html=True)


# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run train_model.ipynb "
        "first to generate diabetes_model.pkl and scaler.pkl."
    )
    st.stop()


# Application heading
st.title("Diabetes Risk Predictor")
st.markdown(
    "### Machine Learning-Based Health Risk Analysis"
)

st.write(
    "Enter the following health measurements to get a "
    "prediction from the trained Support Vector Machine model."
)

st.divider()


# Input section
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.slider(
        "Glucose Level",
        min_value=0,
        max_value=200,
        value=100
    )

    blood_pressure = st.slider(
        "Blood Pressure",
        min_value=0,
        max_value=150,
        value=70
    )

    skin_thickness = st.slider(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.slider(
        "Insulin Level",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.slider(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.slider(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.slider(
        "Age",
        min_value=1,
        max_value=100,
        value=25
    )


st.divider()


# Prediction section
if st.button("Predict Diabetes Risk", use_container_width=True):

    # Arrange inputs in the same order as the training features
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    # Apply the scaler fitted during model training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get estimated probability for the positive class
    probability = model.predict_proba(input_scaled)[0]

    positive_class_index = list(model.classes_).index(1)
    risk_percentage = probability[positive_class_index] * 100

    st.subheader("Prediction Results")

    st.metric(
        label="Model-estimated probability of diabetes",
        value=f"{risk_percentage:.2f}%"
    )

    st.progress(float(risk_percentage / 100))

    if prediction == 1:
        st.error("The model predicts a positive diabetes result.")
    else:
        st.success("The model predicts a negative diabetes result.")

    if risk_percentage > 70:
        st.warning("High model-estimated probability")
    elif risk_percentage > 40:
        st.warning("Moderate model-estimated probability")
    else:
        st.info("Low model-estimated probability")

    st.caption(
        "This is an educational machine learning prediction, "
        "not a medical diagnosis. The probability and categories "
        "are not clinically validated. Consult a qualified "
        "healthcare professional for medical advice."
    )
