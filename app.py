
import streamlit as st
import pandas as pd
import joblib

# Configure the page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

heart_model = load_model()

# Page styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #1d3045);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    background-color: #e05260;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.65rem 1.5rem;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #c83e4d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# App heading
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the four patient attributes below to generate "
    "a machine learning prediction."
)

st.info(
    "Educational demonstration only. This tool is not a medical "
    "diagnostic system."
)

# Patient inputs
st.subheader("Patient Information")

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=45
)

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Serum Cholesterol (mg/dl)",
    min_value=50,
    max_value=700,
    value=200
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

# Make a prediction
if st.button("Predict Heart Disease", use_container_width=True):

    # Arrange inputs in the same order used during training
    input_data = pd.DataFrame(
        [[age, trestbps, chol, thalach]],
        columns=["age", "trestbps", "chol", "thalach"]
    )

    # Generate the prediction
    prediction = heart_model.predict(input_data)[0]

    # Display the result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "The model predicts that heart disease may be present."
        )
    else:
        st.success(
            "The model predicts no heart disease based on the inputs."
        )

    st.caption(
        "This prediction is based on a trained machine learning model "
        "and must not be used to make medical decisions."
    )

# Footer
st.divider()
st.caption("Machine Learning Fundamentals | Assignment 2")
