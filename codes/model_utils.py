# model_utils.py

from tensorflow.keras.models import load_model
import joblib

def save_model_and_scaler(model, scaler, model_filename, scaler_filename):
    # 모델 저장
    model.save(model_filename)
    # 스케일러 저장
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

def load_model_and_scaler(model_filename, scaler_filename):
    # 모델 로드
    model = load_model(model_filename)
    # 스케일러 로드
    scaler = joblib.load(scaler_filename)
    print(f"Model loaded from {model_filename}")
    print(f"Scaler loaded from {scaler_filename}")
    return model, scaler
