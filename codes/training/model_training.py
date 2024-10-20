# model_training.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector

def create_lstm_autoencoder(seq_length, n_features):
    inputs = Input(shape=(seq_length, n_features))
    encoded = LSTM(64, activation='relu')(inputs)
    decoded = RepeatVector(seq_length)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = Dense(n_features)(decoded)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_model(model, train_data, epochs=50, batch_size=32):
    history = model.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True
    )
    return history