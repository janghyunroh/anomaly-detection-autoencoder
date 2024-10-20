# inference.py

import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_utils import load_model_and_scaler

new_data_path = '../datas/sensor_data.csv'

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

def infer():
    # 저장된 모델과 스케일러 로드
    model, scaler = load_model_and_scaler('../models/lstm_autoencoder.h5', '../models/scaler.pkl')
    
    # 새로운 데이터 로드 및 전처리
    new_data, _ = load_and_preprocess_data(new_data_path)  # 추론할 데이터 파일명
    
    # 시퀀스 생성
    sequence_length = 50  # 학습 시 사용한 시퀀스 길이와 동일해야 함
    sequences = create_sequences(new_data.values, sequence_length)
    
    # 모델 예측
    X_test_pred = model.predict(sequences)
    
    # 재구성 오류 계산 (MSE)
    reconstruction_errors = np.mean(np.square(X_test_pred - sequences), axis=(1,2))
    
    # 임계값 설정 (학습 데이터의 재구성 오류 기반으로 설정해야 함)
    threshold = 0.01  # 예시 값
    
    # 이상치 탐지
    anomalies = reconstruction_errors > threshold
    anomaly_indices = np.where(anomalies)[0]
    
    # 결과 출력
    print(f"총 {len(anomaly_indices)}개의 이상치가 발견되었습니다.")
    for idx in anomaly_indices:
        print(f"이상치 시퀀스 시작 인덱스: {idx}")

if __name__ == '__main__':
    infer()
