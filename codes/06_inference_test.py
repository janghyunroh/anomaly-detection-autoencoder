import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from training.data_preprocessing import load_and_preprocess_data, create_sequences
import joblib

class ModelVisualizer:
    def __init__(self, model_path, scaler_path, sequence_length=100):
        """
        Args:
            model_path: 저장된 모델 경로
            scaler_path: 저장된 스케일러 경로
            sequence_length: 시퀀스 길이
        """
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.sequence_length = sequence_length
        
    def visualize_predictions(self, data, feature_names, sample_idx=0, use_original_scale=True):
        """
        데이터의 실제값과 모델의 예측값을 시각화합니다.
        
        Args:
            data: 시각화할 시퀀스 데이터 (shape: [batch_size, sequence_length, n_features])
            feature_names: 특징 이름 리스트
            sample_idx: 시각화할 샘플의 인덱스
            use_original_scale: 원래 스케일로 변환할지 여부
        """
        # 예측 수행
        predictions = self.model.predict(data)
        
        # 시각화할 샘플 선택
        sample_actual = data[sample_idx]
        sample_pred = predictions[sample_idx]
        
        if use_original_scale:
            # 원래 스케일로 변환
            sample_actual_reshaped = sample_actual.reshape(-1, len(feature_names))
            sample_pred_reshaped = sample_pred.reshape(-1, len(feature_names))
            
            sample_actual = self.scaler.inverse_transform(sample_actual_reshaped).reshape(self.sequence_length, -1)
            sample_pred = self.scaler.inverse_transform(sample_pred_reshaped).reshape(self.sequence_length, -1)
        
        # 서브플롯 생성
        n_features = sample_actual.shape[1]
        fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
        if n_features == 1:
            axes = [axes]
        
        # 각 특징별 시각화
        for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            ax.plot(sample_actual[:, i], label='Actual', marker='o')
            ax.plot(sample_pred[:, i], label='Predicted', marker='x')
            ax.set_title(f'{feature_name} - Actual vs Predicted')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def visualize_reconstruction_error(self, data):
        """
        전체 데이터셋의 재구성 오차를 시각화합니다.
        
        Args:
            data: 평가할 데이터 (shape: [batch_size, sequence_length, n_features])
        """
        # 예측 수행
        predictions = self.model.predict(data)
        
        # 재구성 오차 계산 (정규화된 스케일에서)
        mse = np.mean(np.square(data - predictions), axis=(1,2))
        
        # 재구성 오차 분포 시각화
        plt.figure(figsize=(10, 6))
        sns.histplot(mse, bins=50)
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Count')
        plt.show()
        
        # 시계열로 재구성 오차 시각화
        plt.figure(figsize=(15, 6))
        plt.plot(mse, marker='.')
        plt.title('Reconstruction Error Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()
        
        return mse
    
    def visualize_anomaly_threshold(self, data, threshold):
        """
        재구성 오차와 이상치 임계값을 함께 시각화합니다.
        
        Args:
            data: 평가할 데이터
            threshold: 이상치 판단 임계값
        """
        mse = self.visualize_reconstruction_error(data)
        
        plt.figure(figsize=(15, 6))
        plt.plot(mse, label='Reconstruction Error', marker='.')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        plt.title('Reconstruction Error with Anomaly Threshold')
        plt.xlabel('Sample Index')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 이상치 비율 계산 및 출력
        anomalies = mse > threshold
        anomaly_ratio = np.mean(anomalies) * 100
        print(f"Anomaly ratio: {anomaly_ratio:.2f}%")
        
        # 이상치 시점 출력
        anomaly_indices = np.where(anomalies)[0]
        print("\nAnomaly timestamps:")
        for idx in anomaly_indices[:5]:  # 처음 5개만 출력
            print(f"Sample index: {idx}")
        if len(anomaly_indices) > 5:
            print("...")
            
        return anomalies

def main():
    # 데이터 로드 및 전처리
    data_path = '../datas/raw/rpm_1200/1200-1.csv'
    feature_columns = ['acc_x', 'acc_y', 'acc_z']
    
    # 데이터 준비
    normalized_data, scaler = load_and_preprocess_data(data_path, feature_columns)
    sequences = create_sequences(normalized_data.values, 100)
    
    # 스케일러 저장
    joblib.dump(scaler, '../models/vibration_scaler.pkl')
    
    # 모델 시각화 도구 초기화
    visualizer = ModelVisualizer(
        '../models/vibration_model.h5',
        '../models/vibration_scaler.pkl',
        sequence_length=100
    )
    
    # 예측 결과 시각화 (원래 스케일로 변환하여 표시)
    visualizer.visualize_predictions(sequences, feature_columns, use_original_scale=True)
    
    # 재구성 오차 시각화
    visualizer.visualize_reconstruction_error(sequences)
    
    # 이상치 임계값과 함께 시각화
    threshold = 0.1
    anomalies = visualizer.visualize_anomaly_threshold(sequences, threshold)

if __name__ == '__main__':
    main()