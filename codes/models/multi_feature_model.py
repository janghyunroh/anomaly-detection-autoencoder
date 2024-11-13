# models/multi_feature_model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector

def create_multi_feature_model(seq_length, n_features):
    inputs = Input(shape=(seq_length, n_features))
    encoded = LSTM(128, activation='relu')(inputs)
    decoded = RepeatVector(seq_length)(encoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    decoded = Dense(n_features)(decoded)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

