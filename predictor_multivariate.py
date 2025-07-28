import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def prepare_multivariate_data(data_matrix, lookback=30):
    scalers = []
    data_scaled = []

    # Scale each feature independently
    for i in range(data_matrix.shape[1]):
        scaler = MinMaxScaler()
        scaled_col = scaler.fit_transform(data_matrix[:, i].reshape(-1, 1)).flatten()
        data_scaled.append(scaled_col)
        scalers.append(scaler)

    data_scaled = np.stack(data_scaled, axis=-1)

    X, y = [], []
    for i in range(len(data_scaled) - lookback):
        X.append(data_scaled[i:i + lookback])
        y.append(data_scaled[i + lookback])
    return np.array(X), np.array(y), scalers

def build_multivariate_lstm_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_multivariate_predictor(data_matrix, lookback=30, epochs=15):
    X, y, scalers = prepare_multivariate_data(data_matrix, lookback)
    model = build_multivariate_lstm_model((X.shape[1], X.shape[2]), y.shape[1])
    model.fit(X, y, epochs=epochs, verbose=1)
    predictions = model.predict(X)
    return model, predictions, scalers