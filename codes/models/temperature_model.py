# models/temperature_model.py
def create_temperature_model(seq_length, n_features=1):
    inputs = Input(shape=(seq_length, n_features))
    encoded = LSTM(64, activation='relu')(inputs)
    decoded = RepeatVector(seq_length)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = Dense(n_features)(decoded)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder