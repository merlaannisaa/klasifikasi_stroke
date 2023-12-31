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

# Set page config
st.set_page_config(layout="wide")

st.title("Klasifikasi Stroke")

# Submenu untuk memilih halaman
menu = st.sidebar.radio("Navigation", ["Visualisasi", "Klasifikasi"])
model = load_model()
if menu == "Visualisasi":
    # st.subheader("Visualisasi")
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
    st.write(f"Number of samples in Training Data: {len(resX_train)}")
    st.write(f"Number of samples in Testing Data: {len(X_test)}")

    # Mengubah threshold menjadi 0.1
    threshold = 0.1

    # Mengambil probabilitas prediksi
    y_prob = model.predict_proba(X_test)[:, 1]

    # Menggunakan threshold untuk membuat prediksi berdasarkan probabilitas
    y_pred = (y_prob > threshold).astype(int)
    
    # Menghitung akurasi model
    accuracy = accuracy_score(y_test, y_pred)

    # Menampilkan confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(f"### Confusion Matrix")
    # Menggunakan plotly untuk membuat confusion matrix
    fig = ff.create_annotated_heatmap(conf_matrix, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale='Viridis')
    fig.update_layout(width=500, height=400)
    st.plotly_chart(fig)

    # Menghitung recall, F1 score, dan ROC score
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred)
    
    st.write("## Evaluation Metrics on Test Data")
    st.write("### Accuracy:", accuracy)
    st.write("### Recall:", recall)
    st.write("### F1 Score:", f1)
    st.write("### ROC AUC:", roc_score)
    
    metrics = ["Accuracy", "Recall", "F1 Score"]
    values = [accuracy, recall, f1]

    plt.figure(figsize=(20,12))
    plt.bar(metrics, values)
    plt.title("Evaluation Metrics on Test Data")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    st.pyplot(plt)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Membuat kurva ROC
    fig, ax = plt.subplots(figsize=(20, 12), dpi=500)
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_score:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    # Menampilkan kurva ROC di aplikasi Streamlit
    st.pyplot(fig)

elif menu ==  "Klasifikasi":
    st.subheader("Input Data")

    #membagi kolom
    col1, col2 = st.columns(2)

    with col1 :
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender = 0 if gender == "Female" else 1

    with col2 :
        age = st.text_input("Age")

    with col1 :
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        hypertension = 0 if hypertension == "No" else 1

    with col2 :
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        heart_disease = 0 if heart_disease == "No" else 1

    with col1 :
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        ever_married = 0 if ever_married == "No" else 1

    with col2:
        work_type_dict = {"Govt Job": 0, "Never Worked": 1, "Private": 2, "Self-employed": 3, "Children": 4}
        work_type = st.selectbox("Work Type", list(work_type_dict.keys()))
        work_type = work_type_dict[work_type]

    with col1 :
        residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
        residence_type = 0 if residence_type == "Rural" else 1

    with col2:
        avg_glucose_level = st.text_input("Average Glucose Level")

    with col1 :
        bmi = st.text_input("BMI")

    with col2:
        smoking_status_dict = {"Unknown": 0, "Formerly Smoked": 1, "Never Smoked": 2, "Smokes": 3}
        smoking_status = st.selectbox("Smoking Status", list(smoking_status_dict.keys()))
        smoking_status = smoking_status_dict[smoking_status]
    
    if st.button("Predict"):
            try:
                # ... (input processing)
                input_data = [[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]]
                #         # Buat kamus data dengan feature_names
                prediction = predict_stroke(model, input_data)
                
                st.write("## Prediction Result")
                if prediction[0] == 1:
                    st.error("Risiko stroke tinggi!")
                else:
                    st.success("Risiko stroke rendah!")

            except ValueError:
                st.error("Invalid input.")


# Tambahkan kode berikut untuk meng-host aplikasi di Streamlit Sharing
# if __name__ == "__main__":
    # st.write("""
    # # Stroke Prediction App
    # This app predicts the risk of stroke based on patient information.
    # Please fill in the details on the left and click the 'Predict' button.
    # """)
