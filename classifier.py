import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

def build_and_train_classifier(X, y, epochs=15, lookback=20):
    num_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM input: (samples, timesteps, features)
    n_samples, n_timesteps = X_scaled.shape
    X_seq = np.array([
        X_scaled[i:i+lookback] for i in range(n_samples - lookback)
    ])
    y_seq = y_cat[lookback:]

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(64, input_shape=(lookback, 1)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    return model, (X_test, y_test)