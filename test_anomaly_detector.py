import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anomaly_detector import detect_anomalies

# Simulate normal signals
np.random.seed(42)
timesteps = 500
signal1 = np.sin(np.linspace(0, 20, timesteps)) + 0.1 * np.random.randn(timesteps)
signal2 = np.cos(np.linspace(0, 20, timesteps)) + 0.1 * np.random.randn(timesteps)
df = pd.DataFrame({'signal1': signal1, 'signal2': signal2})

# Inject anomalies
df.iloc[200:210] += 3
df.iloc[400:405] -= 3

# Detect anomalies
anomalies, scores, threshold = detect_anomalies(df.values, lookback=30, epochs=20)

# Plot
anomaly_indices = np.where(anomalies)[0]

plt.figure(figsize=(12, 5))
plt.plot(df['signal1'].values[30:], label='Signal 1')
plt.scatter(anomaly_indices, df['signal1'].values[30:][anomaly_indices], color='red', label='Anomalies')
plt.title("Anomaly Detection with LSTM Autoencoder")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()