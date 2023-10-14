import streamlit as st
import pickle

# Fungsi untuk memuat model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk melakukan prediksi
def predict_stroke(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Muat model
model = load_model()

st.title("Stroke Prediction App")

# Input data
st.subheader("Enter Patient Information")

gender = st.selectbox("Gender", ["female", "male"])
gender = 0 if gender == "female" else 1

age = st.text_input("Age")

hypertension = st.selectbox("Hypertension", ["no", "yes"])
hypertension = 0 if hypertension == "no" else 1

heart_disease = st.selectbox("Heart Disease", ["no", "yes"])
heart_disease = 0 if heart_disease == "no" else 1

ever_married = st.selectbox("Ever Married", ["No", "Yes"])
ever_married = 0 if ever_married == "No" else 1

work_type = st.selectbox("Work Type", ["Govt Job", "Never Worked", "Private", "Self-employed", "Children"])
work_type = ["Govt_job", "Never_worked", "Private", "Self-employed", "children"].index(work_type)
esidence_type = st.selectbox("Residence Type", ["rural", "urban"])
residence_type = 0 if residence_type == "rural" else 1

avg_glucose_level = st.text_input("Average Glucose Level")

bmi = st.text_input("BMI")

smoking_status = st.selectbox("Smoking Status", ["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"])
smoking_status = ["Unknown", "formerly_smoked", "never_smoked", "smokes"].index(smoking_status)

if st.button("Predict"):
    input_data = [[gender, int(age), hypertension, heart_disease, ever_married, work_type, residence_type, float(avg_glucose_level), float(bmi), smoking_status]]
    prediction = predict_stroke(model, input_data)
    if prediction[0] == 1:
        st.error("High risk of stroke!")
    else:
        st.success("Low risk of stroke!")
# Tambahkan kode berikut untuk meng-host aplikasi di Streamlit Sharing
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.write("""
    # Stroke Prediction App
    This app predicts the risk of stroke based on patient information.
    Please fill in the details on the left and click the 'Predict' button.
    """)
