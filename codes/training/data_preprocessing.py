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
def create_sequences(data_list, seq_length, stride=1):
    """
    시퀀스를 생성하는 함수로, stride를 적용하여 겹침 간격을 조정할 수 있음.
    
    Parameters:
        data_list (list of np.array): 여러 데이터 배열이 포함된 리스트.
        seq_length (int): 생성할 시퀀스의 길이.
        stride (int): 시퀀스 시작점 간의 간격. 기본값은 1.
    
    Returns:
        np.array: 생성된 시퀀스 배열.
    """
    all_sequences = []
    for data in data_list:
        sequences = []
        for i in range(0, len(data) - seq_length + 1, stride):  # stride 적용
            seq = data[i:i+seq_length]
            sequences.append(seq)
        all_sequences.extend(sequences)
    return np.array(all_sequences)

