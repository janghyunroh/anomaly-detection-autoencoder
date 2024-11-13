import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import fbeta_score, confusion_matrix
import os
import joblib
from utils.config import CONFIG, REQUIRED_FEATURES, PER_MODEL

def create_sequences_with_indices(feature_data, seq_length):
    """
    각 시퀀스가 원본 데이터의 어느 인덱스를 포함하는지 추적하면서 시퀀스 생성
    """
    sequences = []
    sequence_indices = []  # 각 시퀀스에 포함된 원본 데이터의 인덱스들
    
    for i in range(len(feature_data) - seq_length + 1):
        seq = feature_data[i:i+seq_length]
        sequences.append(seq)
        sequence_indices.append(list(range(i, i+seq_length)))
    
    return np.array(sequences), sequence_indices

def prepare_test_data(df, feature_columns, seq_length, scaler):
    """
    테스트 데이터 준비 및 시퀀스 생성
    """
    # 특징 데이터와 레이블 분리
    if isinstance(feature_columns, list):
        feature_data = df[feature_columns].values
        scaled_data = scaler.transform(feature_data)
    else:
        feature_data = df[feature_columns].values.reshape(-1, 1)
        scaled_data = scaler.transform(feature_data)
    
    # 시퀀스 생성 (각 시퀀스가 어떤 원본 데이터 포인트를 포함하는지 추적)
    sequences, sequence_indices = create_sequences_with_indices(scaled_data, seq_length)
    
    return sequences, sequence_indices, df['label'].values

# ================================================================================================= #


def find_anomalous_segments(errors, threshold, min_segment_length=20):
    """
    연속된 이상치 구간 찾기
    """
    anomalies = np.array(errors) > threshold
    anomalous_segments = []
    current_segment = None
    
    for i, anomaly in enumerate(anomalies):
        if anomaly:  # 이상치일 경우
            if current_segment is None:
                current_segment = [i]  # 새로운 구간 시작
            else:
                current_segment.append(i)  # 현재 구간에 추가
        else:  # 정상일 경우
            if current_segment is not None:
                # 구간이 일정 길이를 넘으면 저장
                if len(current_segment) >= min_segment_length:
                    anomalous_segments.append(current_segment)
                current_segment = None

    # 마지막 구간 체크
    if current_segment is not None and len(current_segment) >= min_segment_length:
        anomalous_segments.append(current_segment)
    
    return anomalous_segments


def evaluate_point_wise_anomalies(model, sequences, sequence_indices, n_points, threshold=0.1, ensemble_method='mean', min_segment_length=20):
    """
    데이터 포인트 단위로 이상치 여부 평가
    
    Args:
        model: 학습된 모델
        sequences: 입력 시퀀스들
        sequence_indices: 각 시퀀스가 포함하는 원본 데이터의 인덱스
        n_points: 원본 데이터의 총 개수
        threshold: 이상치 판단 임계값
        ensemble_method: 앙상블 방법 ('mean', 'median', 'majority_vote')
    """
    predicted_anomalies = np.zeros(n_points)

    # 각 시퀀스의 재구성 오차 계산
    predictions = model.predict(sequences)
    sequence_errors = np.mean(np.power(sequences - predictions, 2), axis=2)  # shape: (n_sequences, seq_length)
    # = 시퀀스 별 재구성 오차


    # 각 데이터 포인트별 재구성 오차를 저장할 리스트
    point_errors = [[] for _ in range(n_points)]
    
    # 각 시퀀스의 재구성 오차를 해당되는 데이터 포인트에 할당
    for seq_idx, point_indices in enumerate(sequence_indices):
        for time_step, point_idx in enumerate(point_indices):
            point_errors[point_idx].append(sequence_errors[seq_idx, time_step])
    
    # 각 데이터 포인트별로 최종 이상치 점수 계산
    anomaly_scores = np.zeros(n_points)
    for i in range(n_points):
        if ensemble_method == 'mean':
            anomaly_scores[i] = np.mean(point_errors[i])
        elif ensemble_method == 'median':
            anomaly_scores[i] = np.median(point_errors[i])
        # elif ensemble_method == 'majority_vote':
        #     # 각 시퀀스에서의 이상치 판정을 투표
        #     votes = np.array(point_errors[i]) > threshold
        #     anomaly_scores[i] = np.mean(votes)
    
    # 연속된 이상치 구간 찾기
    anomalous_segments = find_anomalous_segments(anomaly_scores, threshold, min_segment_length)
    
    # 최종 이상치 판정
    # if ensemble_method == 'majority_vote':
    #     predicted_anomalies = (anomaly_scores > 0.5).astype(int)  # 과반수 이상이 이상치로 판단
    # else:
    #     predicted_anomalies = (anomaly_scores > threshold).astype(int)

    for segment in anomalous_segments:
        predicted_anomalies[segment] = 1  # 해당 구간은 이상치로 설정
    
    return predicted_anomalies, anomaly_scores

def evaluate_rpm_models(rpm_config, rpm_type):
    """
    특정 RPM에 대한 모든 모델 평가
    """
    test_data_dir = rpm_config['test_data_dir']
    results_dir = rpm_config['results_dir']
    
    # results_dir이 없으면 생성
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 기존 결과 파일 제거
    print("\nCleaning up previous results...")
    for file_name in os.listdir(results_dir):
        if file_name.startswith('results_'):
            file_path = os.path.join(results_dir, file_name)
            os.remove(file_path)
            print(f"Removed: {file_path}")
    
    for model_type in CONFIG['model_type']:
        if model_type not in PER_MODEL:  # multi-feature 제외
            continue
            
        print(f"\nEvaluating {model_type} model for {rpm_type}:")
        feature_columns = REQUIRED_FEATURES[model_type]
        
        # 모델과 스케일러 로드
        model_path = os.path.join(rpm_config['models_dir'], f'{model_type}_model.h5')
        scaler_path = os.path.join(rpm_config['models_dir'], f'{model_type}_scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Model or scaler not found for {model_type}")
            continue
            
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        all_results = []
        
        # 모든 테스트 파일 평가
        for file_name in os.listdir(test_data_dir):
            if file_name.startswith('anomalous_'+model_type):
                # 테스트 데이터 로드
                file_path = os.path.join(test_data_dir, file_name)
                print(f"Evaluating test file: {file_path}")
                df = pd.read_csv(file_path)
                
                # 시퀀스 데이터 준비
                sequences, sequence_indices, true_labels = prepare_test_data(
                    df,
                    feature_columns,
                    rpm_config['sequence_length'],
                    scaler
                )
                
                # 데이터 포인트 단위 이상치 탐지
                predicted_anomalies, anomaly_scores = evaluate_point_wise_anomalies(
                    model,
                    sequences,
                    sequence_indices,
                    len(df),
                    rpm_config['anomaly_threshold'],
                    ensemble_method='mean'  # 또는 'median', 'majority_vote'
                )
                
                # 성능 평가
                #f1 = fbeta_score(true_labels, predicted_anomalies, beta=1)
                f2 = fbeta_score(true_labels, predicted_anomalies, beta=2)
                conf_matrix = confusion_matrix(true_labels, predicted_anomalies)
                
                all_results.append(f2)
                
                print(f"\nResults for {file_name}:")
                print(f"F2 Score: {f2:.4f}")
                print("Confusion Matrix:")
                print(conf_matrix)
                
                # 결과를 파일로 저장 (선택사항)
                results_dict = pd.DataFrame({
                    'timestamp': df['timestamp'],
                    'true_label': true_labels,
                    'predicted_anomaly': predicted_anomalies,
                    'anomaly_score': anomaly_scores
                })
                
                
                # feature 값들 추가
                if isinstance(feature_columns, list):
                    for feature in feature_columns:
                        results_dict[feature] = df[feature]
                else:
                    results_dict[feature_columns] = df[feature_columns]
                
                # DataFrame 생성 및 저장
                results_df = pd.DataFrame(results_dict)
                
                # 컬럼 순서 조정 (timestamp, features, labels, predictions 순)
                feature_cols = [feature_columns] if isinstance(feature_columns, str) else feature_columns
                column_order = ['timestamp'] + feature_cols + ['true_label', 'predicted_anomaly', 'anomaly_score']
                results_df = results_df[column_order]
                
                # 결과 저장
                results_path = os.path.join(rpm_config['results_dir'], 
                                          f'results_{model_type}_{file_name}')
                results_df.to_csv(results_path, index=False)
                
                print(f"Results saved to: {results_path}")
        
        # 평균 성능 출력
        if all_results:
            avg_f2 = np.mean(all_results)
            print(f"\nAverage F2 Score for {model_type}: {avg_f2:.4f}")

def main():
    # RPM 1200과 600에 대해 각각 평가
    for rpm_type in ['rpm_600']:
        print(f"\n{'='*50}")
        print(f"Evaluating models for {rpm_type}")
        print(f"{'='*50}")
        
        rpm_config = CONFIG[rpm_type]
        evaluate_rpm_models(rpm_config, rpm_type)

if __name__ == "__main__":
    main()