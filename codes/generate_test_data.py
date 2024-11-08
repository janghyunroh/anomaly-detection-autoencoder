# generate_test_data.py
import pandas as pd
import numpy as np
import os

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 이제 directory2의 file2.py를 import할 수 있습니다.
from utils.config import CONFIG, REQUIRED_FEATURES, PER_MODEL

def inject_anomalies_to_session(df, feature_columns, anomaly_ratio=0.1, 
                              min_anomaly_length=5, max_anomaly_length=20):
    """
    더 현실적인 이상치 패턴을 주입하는 함수
    
    Parameters:
    - df: 원본 DataFrame
    - feature_columns: 이상치를 주입할 특징 컬럼들
    - anomaly_ratio: 전체 데이터 중 이상치 비율
    - min_anomaly_length: 최소 이상치 연속 길이
    - max_anomaly_length: 최대 이상치 연속 길이
    """
    anomalous_df = df.copy()
    data_length = len(df)
    anomalous_df['label'] = 0
    
    # 전체 이상치 길이 계산
    total_anomaly_points = int(data_length * anomaly_ratio)
    current_anomaly_points = 0
    
    while current_anomaly_points < total_anomaly_points:
        # 이상치 시작 위치 선택
        start_idx = np.random.randint(0, data_length - max_anomaly_length)
        
        # 이상치 길이 선택
        anomaly_length = np.random.randint(min_anomaly_length, max_anomaly_length)
        
        # 이상치 패턴 선택
        pattern = np.random.choice(['spike', 'trend', 'zero', 'noise'])
        
        # 선택된 특징들에 대해 이상치 주입
        if isinstance(feature_columns, list):
            features = feature_columns
        else:
            features = [feature_columns]
            
        for col in features:
            original_values = anomalous_df.loc[start_idx:start_idx+anomaly_length-1, col].values
            
            if pattern == 'spike':
                # 급격한 스파이크 생성
                spike_factor = np.random.uniform(3, 8)
                anomalous_values = original_values * spike_factor
                
            elif pattern == 'trend':
                # 점진적 증가 또는 감소 트렌드
                trend_factor = np.random.uniform(0.5, 2.0)
                trend = np.linspace(1, trend_factor, anomaly_length)
                anomalous_values = original_values * trend
                
            elif pattern == 'zero':
                # 값이 0으로 떨어지는 현상
                anomalous_values = np.zeros_like(original_values)
                
            else:  # noise
                # 노이즈 증가
                noise = np.random.normal(0, np.std(original_values), anomaly_length)
                anomalous_values = original_values + noise
            
            # 이상치 값 적용
            anomalous_df.loc[start_idx:start_idx+anomaly_length-1, col] = anomalous_values
            
        # 레이블 설정
        anomalous_df.loc[start_idx:start_idx+anomaly_length-1, 'label'] = 1
        
        current_anomaly_points += anomaly_length
    
    return anomalous_df

def inject_multi_feature_anomalies(df, feature_groups, anomaly_ratio=0.1):
    """
    여러 특징 간의 상관관계를 고려한 이상치 주입
    
    Parameters:
    - df: 원본 DataFrame
    - feature_groups: 상관관계가 있는 특징들의 그룹 (예: [['accel_x', 'accel_y', 'accel_z'], ['voltage', 'temperature']])
    """
    anomalous_df = df.copy()
    data_length = len(df)
    anomalous_df['label'] = 0
    
    for feature_group in feature_groups:
        # 각 특징 그룹별로 상관된 이상치 주입
        n_anomalies = int(data_length * anomaly_ratio / len(feature_groups))
        
        for _ in range(n_anomalies):
            # 이상치 시작 위치와 길이 선택
            start_idx = np.random.randint(0, data_length - 20)
            length = np.random.randint(5, 20)
            
            # 그룹 내 모든 특징에 대해 상관된 이상치 생성
            pattern = np.random.choice(['correlated_spike', 'correlated_trend'])
            
            if pattern == 'correlated_spike':
                # 모든 특징이 동시에 비정상적인 값을 보이는 경우
                for feature in feature_group:
                    spike_factor = np.random.uniform(3, 8)
                    original_values = anomalous_df.loc[start_idx:start_idx+length-1, feature].values
                    anomalous_df.loc[start_idx:start_idx+length-1, feature] = original_values * spike_factor
                    
            else:  # correlated_trend
                # 특징들 간의 관계가 비정상적으로 변하는 경우
                base_trend = np.linspace(1, np.random.uniform(1.5, 2.5), length)
                for i, feature in enumerate(feature_group):
                    # 각 특징마다 약간씩 다른 트렌드 적용
                    trend_variation = base_trend * np.random.uniform(0.8, 1.2)
                    original_values = anomalous_df.loc[start_idx:start_idx+length-1, feature].values
                    anomalous_df.loc[start_idx:start_idx+length-1, feature] = original_values * trend_variation
            
            # 레이블 설정
            anomalous_df.loc[start_idx:start_idx+length-1, 'label'] = 1
    
    return anomalous_df

def process_rpm_data(rpm_config, model_type):
    """
    특정 RPM에 대한 테스트 데이터 처리
    """
    test_dir = rpm_config['test_data_dir']
    processed_dir = rpm_config['processed_data_dir']
    
    # 처리된 데이터 저장을 위한 디렉토리 생성
    os.makedirs(processed_dir, exist_ok=True)
    
    feature_columns = REQUIRED_FEATURES[model_type]
    
    # 테스트 디렉토리의 모든 CSV 파일 처리
    for file_name in os.listdir(test_dir):
        if file_name.endswith('.csv'):
            # 원본 데이터 로드
            file_path = os.path.join(test_dir, file_name)
            df = pd.read_csv(file_path)
            
            # 이상치 주입 및 레이블링
            anomalous_df = inject_anomalies_to_session(
                df, 
                feature_columns,
                anomaly_ratio=0.1, 
                min_anomaly_length=40,
                max_anomaly_length=100
            )
            
            # 처리된 데이터 저장
            output_path = os.path.join(processed_dir, f'anomalous_{model_type}_{file_name}')
            anomalous_df.to_csv(output_path, index=False)
            
            print(f"Processed {file_name} for {model_type} model")

def main():
    # RPM 1200과 600에 대해 각각 처리
    for rpm_type in ['rpm_1200', 'rpm_600']:
        rpm_config = CONFIG[rpm_type]
        print(f"\nProcessing {rpm_type}:")
        
        # 각 모델 타입별로 처리
        for model_type in CONFIG['model_type']:
            if model_type in PER_MODEL:  # multi-feature 제외
                print(f"\nGenerating test data for {model_type} model:")
                process_rpm_data(rpm_config, model_type)

if __name__ == "__main__":
    main()