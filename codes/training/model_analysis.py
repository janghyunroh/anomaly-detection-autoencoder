import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from utils.data_loader import load_data
from utils.config import CONFIG, REQUIRED_FEATURES
from model_utils import load_model_and_scaler
from training.data_preprocessing import load_and_preprocess_data, create_sequences


def analyze_model(model_path, scaler_path, rpm):
    """
    학습된 모델의 성능을 분석하는 함수
    
    Args:
        model_path: 학습된 모델 파일 경로
        scaler_path: 저장된 스케일러 파일 경로
        rpm: 분석할 RPM (모델이 학습된 RPM과 동일해야 함)
    """
    # 1. 저장된 모델과 스케일러 로드
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    # 2. 해당 RPM의 데이터 로드
    data = load_data(CONFIG['data_dir'], rpm)
    
    # 3. 정상/이상 데이터 분리
    normal_indices = data['label'] == 0  # 정상 데이터 인덱스
    anomaly_indices = data['label'] == 1  # 이상 데이터 인덱스
    
    normal_data = data[normal_indices]
    anomaly_data = data[anomaly_indices]
    
    # 4. 데이터 전처리 및 시퀀스 생성
    normal_sequences, _ = load_and_preprocess_data(normal_data, REQUIRED_FEATURES['vibration'])
    anomaly_sequences, _ = load_and_preprocess_data(anomaly_data, REQUIRED_FEATURES['vibration'])
    
    normal_sequences = create_sequences(normal_sequences, CONFIG['sequence_length'])
    anomaly_sequences = create_sequences(anomaly_sequences, CONFIG['sequence_length'])
    
    print(f"\n{rpm}RPM 데이터 분석:")
    print(f"정상 시퀀스 수: {len(normal_sequences)}")
    print(f"이상 시퀀스 수: {len(anomaly_sequences)}")
    
    # 5. 재구성 오차 계산
    normal_pred = model.predict(normal_sequences)
    anomaly_pred = model.predict(anomaly_sequences)
    
    normal_errors = np.mean(np.square(normal_sequences - normal_pred), axis=(1,2))
    anomaly_errors = np.mean(np.square(anomaly_sequences - anomaly_pred), axis=(1,2))
    
    # 6. 재구성 오차 분포 시각화
    plt.figure(figsize=(12, 6))
    plt.hist(normal_errors, bins=50, alpha=0.5, label=f'{rpm}rpm Normal', density=True)
    plt.hist(anomaly_errors, bins=50, alpha=0.5, label=f'{rpm}rpm Anomaly', density=True)
    plt.legend()
    plt.title(f'Reconstruction Error Distribution ({rpm}RPM)')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.show()
    
    # 나머지 코드는 동일...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델 파일 경로 (.h5)')
    parser.add_argument('--scaler_path', type=str, required=True,
                        help='저장된 스케일러 파일 경로 (.pkl)')
    parser.add_argument('--rpm', type=int, required=True,
                        help='분석할 RPM (모델이 학습된 RPM과 동일해야 함)')
    
    args = parser.parse_args()
    
    # RPM에 따른 모델 평가
    best_threshold, best_f1 = analyze_model(
        args.model_path,
        args.scaler_path,
        args.rpm
    )