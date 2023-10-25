import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
from sklearn.metrics import recall_score, f1_score, roc_curve, roc_auc_score, precision_score
from sklearn.metrics import classification_report, silhouette_score
import seaborn as sns
import base64

# Fungsi untuk memuat model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# # Fungsi untuk melakukan prediksi
# def predict_stroke(model, input_data):
#     prediction = model.predict(input_data)
#     return prediction

model = load_model()
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

# Mengubah threshold menjadi 0.1
threshold = 0.1

# Mengambil probabilitas prediksi
y_prob = model.predict_proba(X_test)[:, 1]

# Menggunakan threshold untuk membuat prediksi berdasarkan probabilitas
y_pred = (y_prob > threshold).astype(int)

# Set page config
# st.set_page_config(layout="centered")

st.title("Klasifikasi Stroke")

# Submenu untuk memilih halaman
menu = st.sidebar.selectbox("Menu", ["Visualisasi", "Klasifikasi"])
if menu == "Visualisasi":
    # st.subheader("Visualisasi")
    # Mencetak jumlah data dalam set pelatihan dan pengujian
    st.write(f"Number of samples in Training Data: {len(resX_train)}")
    st.write(f"Number of samples in Testing Data: {len(X_test)}")
    
    # Menghitung akurasi model
    accuracy = accuracy_score(y_test, y_pred)

    # Mengatur urutan kelas
    class_order = [1, 0]  # Urutan yang diinginkan, [1, 0] dalam hal ini

    confusion = confusion_matrix(y_test, y_pred, labels=class_order)

    # Menampilkan confusion matrix
    st.write("## Confusion Matrix")
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_order, yticklabels=class_order)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

    # Menghitung recall, F1 score, dan ROC score
    recall = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred)
    
    st.write("## Evaluation Metrics on Test Data")
    st.write("### Accuracy:", accuracy)
    st.write("### Recall:", recall)
    st.write("### Precision:", prec)
    st.write("### F1 Score:", f1)
    st.write("### ROC AUC:", roc_score)
    
    metrics = ["Accuracy", "Recall","Precision", "F1 Score"]
    values = [accuracy, recall, prec, f1]

    plt.figure(figsize=(20,12))
    plt.bar(metrics, values)
    plt.tick_params(axis='x', labelrotation=0, labelsize= 25)
    plt.title("Evaluation Metrics on Test Data", fontsize=30)
    plt.xlabel("Metrics", fontsize=30)
    plt.ylabel("Values", fontsize=30)
    st.pyplot(plt)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Membuat kurva ROC
    fig, ax = plt.subplots(figsize=(20, 12), dpi=500)
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_score:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=30)
    ax.set_ylabel('True Positive Rate', fontsize=30)
    ax.set_title('ROC Curve', fontsize=30)
    ax.legend(loc="lower right")

    # Menampilkan kurva ROC di aplikasi Streamlit
    st.pyplot(fig)

elif menu ==  "Klasifikasi":
    input_type = st.sidebar.selectbox("Pilih Jenis Input", ["User Input", "File Input"])
    if input_type == "User Input":
        #membagi kolom
        col1, col2 = st.sidebar.columns(2)
    
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
        
        if st.sidebar.button("Predict"):
                try:
                    # ... (input processing)
                    input_data = [[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]]
                    input_df = pd.DataFrame(input_data, columns=["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"])
                                
                    # Menampilkan data input pengguna
                    st.write("### Data Input")
                    st.write("Gender:", "Female" if gender == 0 else "Male")
                    st.write("Age:", age)
                    st.write("Hypertension:", "Yes" if hypertension else "No")  # Menggunakan "Yes" atau "No" untuk menampilkan nilai asli
                    st.write("Heart Disease:", "Yes" if heart_disease else "No")  # Menggunakan "Yes" atau "No" untuk menampilkan nilai asli
                    st.write("Ever Married:", "Yes" if ever_married else "No")  # Menggunakan "Yes" atau "No" untuk menampilkan nilai asli
                    # Mengubah "work type" menjadi teks
                    work_type_mapping = {0: "Govt Job", 1: "Never Worked", 2: "Private", 3: "Self-employed", 4: "Children"}
                    work_type_text = work_type_mapping.get(work_type, "Unknown")
                    st.write("Work Type:", work_type_text)
                    st.write("Residence Type:", "Urban" if residence_type else "Rural")  # Menggunakan "Urban" atau "Rural" untuk menampilkan nilai asli
                    st.write("Average Glucose Level:", avg_glucose_level)
                    st.write("BMI:", bmi)
                    # Mengubah "smoking status" menjadi teks
                    smoking_status_mapping = {0: "Unknown", 1: "Formerly Smoked", 2: "Never Smoked", 3: "Smokes"}
                    smoking_status_text = smoking_status_mapping.get(smoking_status, "Unknown")
                    st.write("Smoking Status:", smoking_status_text)

                    threshold = 0.1
                    proba = model.predict_proba(input_data)[:, 1]
                    prediction = (proba > threshold).astype(int)
                    
                    st.write("### Prediction Result")

                    if prediction[0] == 1:
                        st.error("Pasien Memiliki Risiko Stroke!")
                        # st.write("Probabilitas Risiko Stroke:", proba[0])
                    else:
                        st.success("Pasien Tidak Berisiko Stroke!")
                        # st.write("Probabilitas Risiko Stroke:", proba[0])
                    
                    # # Jika Anda ingin menambahkan hasil prediksi ke dalam DataFrame
                    # input_df["Prediction"] = prediction
                    # st.write("### Data Masukan dengan Hasil Prediksi")
                    # st.write(input_df)
            
                except ValueError:
                    st.error("Invalid input.")
    
    elif input_type == "File Input":
        input_type = st.sidebar.selectbox("Pilih Jenis Input", ["Data without Label", "Data with Label"])
        if input_type == "Data without Label":
            uploaded_file = st.sidebar.file_uploader("Upload File", type=["csv"])

            if uploaded_file is not None:
                st.sidebar.write("Upload File Success")
                file = pd.read_csv(uploaded_file)
                required_columns = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
                missing_columns = [col for col in required_columns if col not in file.columns]
                if missing_columns :
                    st.error(f"Kolom_kolom berikut tidak ditemukan dalam file: {', '.join(missing_columns)}'")
                else:
                    lab_enc = LabelEncoder()      
                    lab_enc_data= file[required_columns]
                    for x in lab_enc_data.columns:
                        lab_enc_data[x]=lab_enc.fit_transform(lab_enc_data[x])
                    for x in lab_enc_data.columns:
                        file[x]=lab_enc_data[x]
                    input_data = file
                    
                    if st.sidebar.button("Predict"):
    
                        threshold = 0.1
                        proba = model.predict_proba(input_data)[:, 1]
                        prediction = (proba > threshold).astype(int)

                        total_data = len(file)
                        jumlah_1 = sum(prediction)
                        jumlah_0 = total_data = jumlah_1
                        st.write("Jumlah Data: {total_data}")
                        st.write("Terprediksi Stroke: {jumlah_1}")
                        st.write("Terprediksi Tidak Stroke: {jumlah_0}")
                        
                        file2["Prediction"] = prediction
                        # file["Probabilitas"] = proba
                            
                        st.write("## Hasil Prediksi Pada Data")
                        st.write(file2)

                        csv = file.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi.csv">Unduh Hasil Prediksi (CSV)</a>'
                        st.markdown(href, unsafe_allow_html=True)

        else:
            uploaded_file = st.sidebar.file_uploader("Upload File", type=["csv"])

            if uploaded_file is not None:
                st.sidebar.write("Upload File Success")
                file = pd.read_csv(uploaded_file)
                file2 = pd.read_csv(uploaded_file)
                X = file.drop(columns=['stroke'])
                Y = file['stroke']
                required_columns = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
                missing_columns = [col for col in required_columns if col not in file.columns]
                
                if missing_columns :
                    st.error(f"Kolom_kolom berikut tidak ditemukan dalam file: {', '.join(missing_columns)}'")
                else:
                    lab_enc = LabelEncoder()      
                    lab_enc_data= X[required_columns]
                    for x in lab_enc_data.columns:
                        lab_enc_data[x]=lab_enc.fit_transform(lab_enc_data[x])
                    for x in lab_enc_data.columns:
                        X[x]=lab_enc_data[x]
                    input_data = X
                    
                    if st.sidebar.button("Predict"):
    
                        threshold = 0.1
                        proba = model.predict_proba(input_data)[:, 1]
                        prediction = (proba > threshold).astype(int)

                        # prediction = model.predict(input_data)   
                        acc = accuracy_score(Y, prediction)
                        recall = recall_score(Y, prediction)
                        prec = precision_score(Y, prediction)
                        f1 = f1_score(Y, prediction)
                        file2["Prediction"] = prediction

                        st.write("Akurasi :", acc)
                        st.write("Recall :", recall)
                        st.write("Precision :", prec)
                        st.write("F1 Score :", f1)
                        st.write("## Hasil Prediksi Pada Data") 
                        st.write(file2)
                              
                        csv = file.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi.csv">Unduh Hasil Prediksi (CSV)</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Mengatur urutan kelas
                        class_order = [1, 0]  # Urutan yang diinginkan, [1, 0] dalam hal ini
                        confusion = confusion_matrix(Y, prediction, labels=class_order)
                        # Menampilkan confusion matrix
                        st.write("## Confusion Matrix")
                        plt.figure(figsize=(6, 4))
                        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_order, yticklabels=class_order)
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        st.pyplot(plt)
