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

# Set page config
st.set_page_config(layout="wide")

# Muat model
model = load_model()

st.title("Stroke Prediction App")

# Input data
st.subheader("Enter Patient Information")

gender = st.radio("Gender", ["Female", "Male"])
gender = 0 if gender == "Female" else 1

age = st.text_input("Age")

hypertension = st.radio("Hypertension", ["No", "Yes"])
hypertension = 0 if hypertension == "No" else 1

heart_disease = st.radio("Heart Disease", ["No", "Yes"])
heart_disease = 0 if heart_disease == "No" else 1

ever_married = st.radio("Ever Married", ["No", "Yes"])
ever_married = 0 if ever_married == "No" else 1

work_type_dict = {"Govt Job": 0, "Never Worked": 1, "Private": 2, "Self-employed": 3, "Children": 4}
work_type = st.radio("Work Type", list(work_type_dict.keys()))
work_type = work_type_dict[work_type]

residence_type = st.radio("Residence Type", ["Rural", "Urban"])
residence_type = 0 if residence_type == "Rural" else 1

avg_glucose_level = st.text_input("Average Glucose Level")

bmi = st.text_input("BMI")

smoking_status_dict = {"Unknown": 0, "Formerly Smoked": 1, "Never Smoked": 2, "Smokes": 3}
smoking_status = st.radio("Smoking Status", list(smoking_status_dict.keys()))
smoking_status = smoking_status_dict[smoking_status]

if st.button("Predict"):
    try:
        age = int(age)
        avg_glucose_level = float(avg_glucose_level)
        bmi = float(bmi)
        input_data = [[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]]
        prediction = predict_stroke(model, input_data)
        
        st.write("## Prediction Result")
        if prediction[0] == 1:
            st.error("High risk of stroke!")
        else:
            st.success("Low risk of stroke!")

        plot_stroke_risk(prediction)
    except ValueError:
        st.error("Invalid input. Please check the values you entered.")

# Fungsi untuk membuat visualisasi
def plot_stroke_risk(prediction):
    labels = ["Low Risk", "High Risk"]
    values = [prediction[0], 1 - prediction[0]]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title("Stroke Risk")

    # Menampilkan gambar Matplotlib di Streamlit
    st.pyplot(fig)


# Tambahkan kode berikut untuk meng-host aplikasi di Streamlit Sharing
if __name__ == "__main__":
    st.write("""
    # Stroke Prediction App
    This app predicts the risk of stroke based on patient information.
    Please fill in the details on the left and click the 'Predict' button.
    """)
