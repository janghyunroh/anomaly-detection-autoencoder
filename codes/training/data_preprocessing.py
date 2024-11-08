# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# feature column 선택해서 로드
def load_and_preprocess_data(df_list, feature_columns):
    
    scaler = MinMaxScaler()
    normalized_data_list = []
    
    for df in df_list:
        # Extract the specified features
        if isinstance(feature_columns, str):
            data = df[[feature_columns]].values
        else:
            data = df[feature_columns].values
        
        # Normalize the data
        normalized_data = scaler.fit_transform(data)
        normalized_data_list.append(normalized_data)
    
    return normalized_data_list, scaler

# session 에서 시퀀스들을 생성
def create_sequences(data_list, seq_length):
    all_sequences = []
    for data in data_list:
        sequences = []
        for i in range(len(data) - seq_length + 1):
            seq = data[i:i+seq_length]
            sequences.append(seq)
        all_sequences.extend(sequences)
    return np.array(all_sequences)
