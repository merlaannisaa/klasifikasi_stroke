import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
from sklearn.metrics import recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report

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

# Fungsi untuk membuat visualisasi akurasi dengan bar chart
def plot_accuracy(accuracy):
    labels = ["Accuracy"]
    values = [accuracy]

    plt.figure(figsize=(5, 5))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    st.pyplot()
    
# Fungsi untuk membuat visualisasi ROC curve
def plot_roc_curve(y_test, y_pred, y_train, y_pred_train):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (Test Data)')
    plt.plot(fpr_train, tpr_train, label='ROC curve (Train Data)', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    st.pyplot()


# Set page config
st.set_page_config(layout="wide")

st.title("Klasifikasi Stroke")

# Submenu untuk memilih halaman
menu = st.sidebar.radio("Navigation", ["Klasifikasi", "Visualisasi"])
model = load_model()
if menu == "Klasifikasi":
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
                # ... (input processing)
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

elif menu == "Visualisasi":
    st.subheader("Visualisasi")
    # Memuat dataset
    df = pd.read_csv('stroke_dataset.csv')
    
    # Melakukan balanced split
    # Memisahkan dataset menjadi dua berdasarkan kelas
    stroke_class = df[df['stroke'] == 1]  # Memisahkan data dengan kelas 1 (stroke)
    non_stroke_class = df[df['stroke'] == 0]  # Memisahkan data dengan kelas 0 (tidak stroke)
    
    # Memisahkan stroke_class menjadi data pelatihan dan pengujian dengan perbandingan 70:30
    X_stroke = stroke_class.drop(columns=['stroke'])
    y_stroke = stroke_class['stroke']
    
    X_train_stroke, X_test_stroke, y_train_stroke, y_test_stroke = train_test_split(X_stroke, y_stroke, test_size=0.3, random_state=0)
    
    X_non_stroke = non_stroke_class.drop(columns=['stroke'])
    y_non_stroke = non_stroke_class['stroke']
    
    # Menggunakan len(X_test_stroke) untuk menentukan jumlah data pengujian yang sama dengan stroke_class
    X_train_non_stroke, X_test_non_stroke, y_train_non_stroke, y_test_non_stroke = train_test_split(
        X_non_stroke, y_non_stroke, test_size=len(X_test_stroke), random_state=0)
    
    # Menggabungkan data pelatihan dan pengujian dari kedua kelas (stroke dan non-stroke)
    X_train = pd.concat([X_train_stroke, X_train_non_stroke])
    y_train = pd.concat([y_train_stroke, y_train_non_stroke])
    
    X_test = pd.concat([X_test_stroke, X_test_non_stroke])
    y_test = pd.concat([y_test_stroke, y_test_non_stroke])
    
    # Melakukan SMOTE-ENN untuk oversampling
    smoteenn = SMOTEENN(random_state=0, smote=SMOTE(sampling_strategy='auto', k_neighbors=36, random_state=0))
    resX_train, resY_train= smoteenn.fit_resample(X_train,y_train)

        # Mencetak jumlah data dalam set pelatihan dan pengujian
    st.write(f"Number of samples in Training Data: {len(X_train)}")
    st.write(f"Number of samples in Testing Data: {len(X_test)}")

    # Mengubah threshold menjadi 0.1
    threshold = 0.1

    # Mengambil probabilitas prediksi
    y_prob = model.predict_proba(X_test)[:, 1]

    # Menggunakan threshold untuk membuat prediksi berdasarkan probabilitas
    y_pred = (y_prob > threshold).astype(int)
    
    # Menghitung akurasi model
    accuracy = accuracy_score(y_test, y_pred)

    # Menampilkan visualisasi akurasi dengan bar chart
    st.write(f"## Model Accuracy on Test Data: {accuracy:.2%}")
    plot_accuracy(accuracy)

    # Menampilkan confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(f"### Confusion Matrix")
    # Menggunakan plotly untuk membuat confusion matrix
    fig = ff.create_annotated_heatmap(conf_matrix, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale='Viridis')
    fig.update_layout(width=500, height=400)
    st.plotly_chart(fig)

    # Menampilkan informasi tambahan: Recall, F1 Score, dan ROC Curve
    classification_report_output = classification_report(y_test_stroke, y_pred)
    st.write(f"### Classification Report")
    st.text(classification_report_output)

    # Menampilkan ROC curve
    st.write("### Receiver Operating Characteristic (ROC) Curve")
    plot_roc_curve(y_test, y_pred, y_train, y_pred_train)

# Tambahkan kode berikut untuk meng-host aplikasi di Streamlit Sharing
if __name__ == "__main__":
    st.write("""
    # Stroke Prediction App
    This app predicts the risk of stroke based on patient information.
    Please fill in the details on the left and click the 'Predict' button.
    """)
