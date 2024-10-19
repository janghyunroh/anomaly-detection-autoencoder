# data_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(csv_file):
    # CSV 파일 로드
    data = pd.read_csv(csv_file)
    
    # 필요한 컬럼 선택
    data = data[['timestamp', 'acc_x', 'acc_y', 'acc_z']]
    
    # timestamp를 시간 순으로 정렬 (필요한 경우)
    data = data.sort_values(by='timestamp')
    
    # 결측치 확인 및 처리
    data = data.dropna()
    
    # 특징 스케일링 (MinMaxScaler)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[['acc_x', 'acc_y', 'acc_z']])
    
    # 스케일링된 데이터를 데이터프레임으로 변환
    scaled_data = pd.DataFrame(scaled_features, columns=['acc_x', 'acc_y', 'acc_z'])
    
    return scaled_data, scaler
