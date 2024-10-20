# train_model.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from data_preprocessing import load_and_preprocess_data
from model_utils import save_model_and_scaler
import argparse
import os


file_path = '../datas/sensor_data.csv'


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

def build_model(seq_length, n_features):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, n_features), return_sequences=False),
        RepeatVector(seq_length),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train(model_save_path, scaler_save_path):
    # 데이터 로드 및 전처리
    data, scaler = load_and_preprocess_data(file_path)
    
    # 시퀀스 생성
    sequence_length = 50  # 시퀀스 길이 설정
    sequences = create_sequences(data.values, sequence_length)
    
    # 학습 데이터 분리
    X_train = sequences
    
    # 모델 구축
    n_features = X_train.shape[2]
    model = build_model(sequence_length, n_features)
    
    # 모델 학습
    model.fit(X_train, X_train, epochs=100, batch_size=32)
    
    # 모델 및 스케일러 저장
    save_model_and_scaler(model, scaler, model_save_path, scaler_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', type=str, default='../models/lstm_autoencoder.h5', help='모델 저장 경로')
    parser.add_argument('--scaler_save_path', type=str, default='../models/scaler.pkl', help='스케일러 저장 경로')
    args = parser.parse_args()
    train(args.model_save_path, args.scaler_save_path)
