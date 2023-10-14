import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk memuat model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk melakukan prediksi
def predict_stroke(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Fungsi untuk membuat visualisasi
def plot_stroke_risk(prediction):
    labels = ["Low Risk", "High Risk"]
    values = [prediction[0], 1 - prediction[0]]

    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Stroke Risk")
    st.pyplot()

# Muat model
model = load_model()

st.title("Stroke Prediction App")

# Input data
st.subheader("Enter Patient Information")

gender = st.radio("Gender", {"Female (0)": 0, "Male (1)": 1})

age = st.text_input("Age")

hypertension = st.radio("Hypertension", {"No (0)": 0, "Yes (1)": 1})

heart_disease = st.radio("Heart Disease", {"No (0)": 0, "Yes (1)": 1})

ever_married = st.radio("Ever Married", {"No (0)": 0, "Yes (1)": 1})

work_type_dict = {"Govt Job (0)": 0, "Never Worked (1)": 1, "Private (2)": 2, "Self-employed (3)": 3, "Children (4)": 4}
work_type = st.radio("Work Type", work_type_dict)

residence_type = st.radio("Residence Type", {"Rural (0)": 0, "Urban (1)": 1})

avg_glucose_level = st.text_input("Average Glucose Level")

bmi = st.text_input("BMI")

smoking_status_dict = {"Unknown (0)": 0, "Formerly Smoked (1)": 1, "Never Smoked (2)": 2, "Smokes (3)": 3}
smoking_status = st.radio("Smoking Status", smoking_status_dict)

if st.button("Predict"):
    input_data = [[gender, int(age), hypertension, heart_disease, ever_married, work_type, residence_type, float(avg_glucose_level), float(bmi), smoking_status]]
    prediction = predict_stroke(model, input_data)
    
    st.write("## Prediction Result")
    if prediction[0] == 1:
        st.error("High risk of stroke!")
    else:
        st.success("Low risk of stroke!")

    plot_stroke_risk(prediction)

# Tambahkan kode berikut untuk meng-host aplikasi di Streamlit Sharing
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.write("""
    # Stroke Prediction App
    This app predicts the risk of stroke based on patient information.
    Please fill in the details on the left and click the 'Predict' button.
    """)
