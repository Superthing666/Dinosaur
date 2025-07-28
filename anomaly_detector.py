import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from sklearn.preprocessing import StandardScaler

def build_autoencoder_model(timesteps, features, latent_dim=32):
    inputs = Input(shape=(timesteps, features))
    encoded = LSTM(latent_dim, activation='relu')(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(features, activation='linear', return_sequences=True)(decoded)
    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

def detect_anomalies(data, lookback=30, epochs=20, threshold=None):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X_seq = np.array([data_scaled[i:i+lookback] for i in range(len(data_scaled) - lookback)])

    model = build_autoencoder_model(lookback, data.shape[1])
    model.fit(X_seq, X_seq, epochs=epochs, batch_size=32, shuffle=True, verbose=0)

    X_pred = model.predict(X_seq)
    mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1, 2))

    if threshold is None:
        threshold = np.percentile(mse, 95)

    anomalies = mse > threshold

    return anomalies, mse, threshold