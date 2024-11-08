# evaluate_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import fbeta_score, confusion_matrix
import os
from utils.config import CONFIG, REQUIRED_FEATURES, PER_MODEL

def prepare_test_sequences(df, feature_columns, seq_length):
    """
    DataFrame을 시퀀스 데이터와 레이블로 변환
    """
    # 특징 데이터와 레이블 분리
    if isinstance(feature_columns, list):
        feature_data = df[feature_columns].values
    else:
        feature_data = df[feature_columns].values.reshape(-1, 1)
    
    labels = df['label'].values
    
    sequences = []
    sequence_labels = []
    
    # 시퀀스 생성
    for i in range(len(feature_data) - seq_length + 1):
        seq = feature_data[i:i+seq_length]
        # 시퀀스 내에 하나라도 이상치가 있으면 해당 시퀀스를 이상으로 레이블링
        is_anomaly = 1 if np.any(labels[i:i+seq_length] == 1) else 0
        sequences.append(seq)
        sequence_labels.append(is_anomaly)
    
    return np.array(sequences), np.array(sequence_labels)

def evaluate_model_performance(model, test_sequences, true_labels, threshold=0.1):
    """
    모델 성능 평가
    """
    # 모델 예측
    predictions = model.predict(test_sequences)
    
    # 재구성 오류 계산
    mse = np.mean(np.power(test_sequences - predictions, 2), axis=(1,2))
    
    # 이상치 판별
    predicted_anomalies = (mse > threshold).astype(int)
    
    # 성능 지표 계산
    f2 = fbeta_score(true_labels, predicted_anomalies, beta=2)
    conf_matrix = confusion_matrix(true_labels, predicted_anomalies)
    
    return {
        'f2_score': f2,
        'confusion_matrix': conf_matrix,
        'mse': mse,
        'predictions': predicted_anomalies
    }

def evaluate_rpm_models(rpm_config, rpm_type):
    """
    특정 RPM에 대한 모든 모델 평가
    """
    processed_dir = rpm_config['processed_data_dir']
    
    # 각 모델 타입별로 평가
    for model_type in CONFIG['model_type']:
        if model_type not in PER_MODEL:  # multi-feature 제외
            continue
            
        print(f"\nEvaluating {model_type} model for {rpm_type}:")
        feature_columns = REQUIRED_FEATURES[model_type]
        
        # 모델 로드
        model_path = os.path.join(rpm_config['models_dir'], f'{model_type}_model.h5')
        if not os.path.exists(model_path):
            continue
        model = tf.keras.models.load_model(model_path)
        
        all_results = []
        
        # 모든 테스트 파일 평가
        for file_name in os.listdir(processed_dir):
            if file_name.startswith('anomalous_'+model_type):
                # 테스트 데이터 로드
                file_path = os.path.join(processed_dir, file_name)
                df = pd.read_csv(file_path)
                
                # 시퀀스 데이터 준비
                sequences, labels = prepare_test_sequences(
                    df,
                    feature_columns,
                    rpm_config['sequence_length']
                )
                
                # 모델 평가
                results = evaluate_model_performance(
                    model,
                    sequences,
                    labels,
                    rpm_config['anomaly_threshold']
                )
                
                all_results.append(results['f2_score'])
                
                print(f"\nSession {file_name}:")
                print(f"F2 Score: {results['f2_score']:.4f}")
                print("Confusion Matrix:")
                print(results['confusion_matrix'])
        
        # 평균 성능 출력
        if all_results:
            avg_f2 = np.mean(all_results)
            print(f"\nAverage F2 Score for {model_type}: {avg_f2:.4f}")

def main():
    # RPM 1200과 600에 대해 각각 평가
    for rpm_type in ['rpm_1200', 'rpm_600']:
        print(f"\n{'='*50}")
        print(f"Evaluating models for {rpm_type}")
        print(f"{'='*50}")
        
        rpm_config = CONFIG[rpm_type]
        evaluate_rpm_models(rpm_config, rpm_type)

if __name__ == "__main__":
    main()