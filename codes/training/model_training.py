# model_training.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector
from utils.config import PER_MODEL
from training.learning_monitor import ReconstructionErrorCallback

def create_lstm_autoencoder(seq_length, model_type):

    n_features = PER_MODEL[model_type]['required_feature_num']
    layer_size = PER_MODEL[model_type]['layer_size']

    inputs = Input(shape=(seq_length, n_features))
    encoded = LSTM(layer_size, activation='relu')(inputs)
    decoded = RepeatVector(seq_length)(encoded)
    decoded = LSTM(layer_size, activation='relu', return_sequences=True)(decoded)
    decoded = Dense(n_features)(decoded)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_model(model, train_data, epochs=50, batch_size=32, validation_split=0.1):
    """
    LSTM AutoEncoder 모델을 학습시키는 함수
    
    Parameters:
    -----------
    model : keras.Model
        학습시킬 LSTM AutoEncoder 모델
    train_data : numpy.ndarray
        학습 데이터. Shape: (samples, sequence_length, features)
    epochs : int, default=50
        전체 데이터셋에 대한 학습 반복 횟수
    batch_size : int, default=32
        한 번에 학습할 데이터 샘플의 개수
    validation_split : float, default=0.1
        검증에 사용할 데이터의 비율 (0.1 = 10%)
    
    Returns:
    --------
    history : keras.callbacks.History
        학습 과정에서의 loss 기록을 담고 있는 객체
        history.history에는 다음 정보가 딕셔너리 형태로 저장됨:
        - 'loss': 각 epoch의 학습 손실값
        - 'val_loss': 각 epoch의 검증 손실값
    """
    recon_error_callback = ReconstructionErrorCallback()
    
    history = model.fit(
        x=train_data,           # 입력 데이터
        y=train_data,           # 타겟 데이터 (AutoEncoder는 입력을 그대로 복원하는 것이 목표)
        epochs=epochs,          # 전체 데이터에 대한 학습 반복 횟수
        batch_size=batch_size,  # 한 번에 처리할 데이터 개수
        validation_split=validation_split,  # 검증 데이터 비율
        shuffle=True,           # 매 epoch마다 데이터 섞기
        verbose=1,              # 학습 진행 상황 출력 (1: 진행바 표시)
        callbacks = [recon_error_callback]
    )
    
    return history