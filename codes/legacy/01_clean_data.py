import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.config import CONFIG, REQUIRED_FEATURES
import numpy as np



def clean_vibration_data(df, features=['accel_x', 'accel_y', 'accel_z'], n_std=3):
    """
    가속도 데이터 노이즈 제거
    
    Parameters:
        df: 원본 데이터프레임
        features: 처리할 가속도 특징들
        n_std: 표준편차의 몇 배를 임계값으로 사용할지
    """
    df_clean = df.copy()
    
    for feature in features:
        # 1. 기본 통계 계산
        mean = df[feature].mean()
        std = df[feature].std()
        
        # 2. 임계값 설정
        upper_limit = mean + n_std * std
        lower_limit = mean - n_std * std
        
        # 3. 이상치를 경계값으로 대체
        df_clean.loc[df_clean[feature] > upper_limit, feature] = upper_limit
        df_clean.loc[df_clean[feature] < lower_limit, feature] = lower_limit
        
        # 결과 출력
        #n_clipped = len(df[feature]) - len(df_clean[
        #    (df_clean[feature] >= lower_limit) & 
        #    (df_clean[feature] <= upper_limit)
        #])
        n_clipped = len(df[(df[feature] > upper_limit) | (df[feature] < lower_limit)])
        print(f"{feature}:")
        print(f"  - 원본 범위: [{df[feature].min():.3f}, {df[feature].max():.3f}]")
        print(f"  - 정제 범위: [{lower_limit:.3f}, {upper_limit:.3f}]")
        print(f"  - 수정된 데이터 포인트: {n_clipped}")
        
    return df_clean

# 시각화 함수
def visualize_cleaning_results(original_df, cleaned_df, feature):
    """
    노이즈 제거 전후 비교 시각화
    """
    plt.figure(figsize=(20, 8))
    
    # 원본 데이터
    plt.subplot(2, 1, 1)
    plt.plot(original_df[feature], 'b-', linewidth=0.1, alpha=0.7, label='Original')
    plt.title('Original Data')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 정제된 데이터
    plt.subplot(2, 1, 2)
    plt.plot(cleaned_df[feature], 'r-', linewidth=0.1, alpha=0.7, label='Cleaned')
    plt.title('Cleaned Data')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 사용 예시
def process_vibration_data(rpm_type='rpm_1200'):
    # 설정 불러오기
    rpm_config = CONFIG[rpm_type]
    raw_dir = rpm_config['raw_data_dir']
    features = REQUIRED_FEATURES['vibration']
    
    for root, dirs, files in os.walk(raw_dir):
        for file_name in files:
            if file_name.endswith('.csv'):
                print(f"\n처리 중인 파일: {file_name}")
                
                # 데이터 로드
                df = pd.read_csv(os.path.join(root, file_name))
                
                # 노이즈 제거
                cleaned_df = clean_vibration_data(df, features)
                
                # 각 축별 결과 시각화
                for feature in features:
                    visualize_cleaning_results(df, cleaned_df, feature)
                
                # 정제된 데이터 저장
                relative_path = os.path.relpath(root, raw_dir)
                processed_path = os.path.join(rpm_config['processed_data_dir'], relative_path)
                processed_file_path = os.path.join(processed_path, f'cleaned_{file_name}')


                os.makedirs(processed_path, exist_ok=True)
                
                cleaned_df.to_csv(processed_file_path, index=False)
                print(f"정제된 데이터 저장됨: {processed_file_path}")

# 실행
#process_vibration_data('rpm_1200')
process_vibration_data('rpm_600')