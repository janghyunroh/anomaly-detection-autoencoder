# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# feature column 선택해서 로드
def load_and_preprocess_data(file_path, feature_column):
    df = pd.read_csv(file_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp')
    
    # Extract the specified feature
    data = df[feature_column].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data, scaler

# session 에서 시퀀스들을 생성
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)
