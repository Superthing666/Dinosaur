import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from classifier import build_and_train_classifier

# Load dataset
df = pd.read_csv("labeled_simulated_timeseries.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# Train classifier
model, (X_test, y_test) = build_and_train_classifier(X, y, epochs=20, lookback=20)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_true_labels, y_pred_labels))