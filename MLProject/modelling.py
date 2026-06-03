import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# 1. Aktifkan MLflow Autolog 
# Fitur ini otomatis mencatat semua parameter, metrik, dan model ke MLflow Tracking UI
mlflow.autolog()

def train_model():
    # 2. Tentukan path dataset bersih
    data_path = "namadataset_preprocessing/data_bersih.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan! Pastikan file data_bersih.csv sudah dimasukkan ke folder yang benar.")
        return

    # 3. Memuat data bersih
    df = pd.read_csv(data_path)
    
    # 4. Memisahkan Fitur (X) dan Target (y)
    # Kolom target kita adalah 'target' (0 = Sehat, 1 = Sakit)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 5. Split data menjadi Training dan Testing (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Mengatur alamat MLflow Tracking UI ke lokalhost
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Heart_Disease_Experiment")
    
    # 7. Memulai proses training di dalam MLflow run
    with mlflow.start_run(run_name="Random_Forest_Baseline"):
        print("Sedang melatih model Random Forest...")
        
        # Inisialisasi model baseline (tanpa tuning hyperparameter karena kriteria basic)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Melakukan evaluasi score sederhana
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"Training Selesai!")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_model()