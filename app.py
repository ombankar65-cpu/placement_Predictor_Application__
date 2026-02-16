import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🎓 Student Placement Predictor")
st.write("Enter student details to predict campus placement status.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    ssc_p = st.number_input("10th (SSC) Percentage", min_value=0.0, max_value=100.0, value=65.0)
    hsc_p = st.number_input("12th (HSC) Percentage", min_value=0.0, max_value=100.0, value=65.0)

with col2:
    hsc_s = st.selectbox("12th Specialization", options=["Commerce", "Science", "Arts"])
    degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=65.0)
    mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=65.0)

# Map categorical data to the numerical format your model likely expects
# Note: If your model used LabelEncoding or One-Hot Encoding, adjust these mappings accordingly
gender_val = 1 if gender == "Male" else 0
hsc_s_map = {"Commerce": 0, "Science": 1, "Arts": 2}
hsc_s_val = hsc_s_map[hsc_s]

# Predict button
if st.button("Predict Placement Status"):
    # Arrange features in the exact order the model was trained on
    features = pd.DataFrame([[gender_val, ssc_p, hsc_p, hsc_s_val, degree_p, mba_p]], 
                            columns=['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'mba_p'])
    
    prediction = model.predict(features)
    
    if prediction[0] == "Placed":
        st.success("🎉 Predicted Status: **Placed**")
    else:
        st.warning("⚠️ Predicted Status: **Not Placed**")

st.divider()
st.info("This model uses a K-Neighbors Classifier to determine placement probability based on academic history.")
