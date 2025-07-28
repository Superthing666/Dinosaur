import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools.predictor_multivariate import train_multivariate_predictor
from classifier import build_and_train_classifier
from anomaly_detector import detect_anomalies
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🧠 Time Pattern Engine v3 – Multivariate Dashboard")

uploaded_file = st.file_uploader("Upload your multivariate signal CSV file", type=["csv"])
mode = st.radio("Select Mode", ["Prediction", "Classification", "Anomaly Detection"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    if mode == "Prediction":
        lookback = st.slider("Lookback Window", min_value=10, max_value=100, value=30)
        epochs = st.slider("Training Epochs", min_value=5, max_value=100, value=20)
        signals = df.columns.tolist()
        st.sidebar.subheader("Signal Selection")
        selected_signals = st.sidebar.multiselect("Choose signals to model", signals, default=signals)

        if len(selected_signals) >= 1:
            data_matrix = df[selected_signals].dropna().to_numpy()

            with st.spinner("Training LSTM model..."):
                model, predictions, scalers = train_multivariate_predictor(data_matrix, lookback=lookback, epochs=epochs)

            st.success("✅ Model trained and predictions generated.")

            aligned_true = data_matrix[lookback:]
            time_axis = np.arange(len(aligned_true))

            for i, signal_name in enumerate(selected_signals):
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(time_axis, aligned_true[:, i], label=f"{signal_name} (Actual)", color='skyblue')
                ax.plot(time_axis, predictions[:, i], label=f"{signal_name} (Predicted)", color='orange')
                ax.set_title(f"Signal: {signal_name}")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Amplitude")
                ax.legend()
                st.pyplot(fig)
        else:
            st.warning("Please select at least one signal column.")

    elif mode == "Classification":
        if "label" not in df.columns:
            st.error("Classification mode requires a 'label' column in the dataset.")
        else:
            X = df.drop("label", axis=1).values
            y = df["label"].values
            with st.spinner("Training classifier..."):
                model, (X_test, y_test) = build_and_train_classifier(X, y, epochs=20, lookback=20)
            st.success("✅ Classifier trained.")

            y_pred = model.predict(X_test)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_test, axis=1)

            st.subheader("Classification Report")
            report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true_labels, y_pred_labels)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

    elif mode == "Anomaly Detection":
        lookback = st.slider("Lookback Window", min_value=10, max_value=100, value=30)
        epochs = st.slider("Training Epochs", min_value=5, max_value=100, value=20)
        signals = df.columns.tolist()
        st.sidebar.subheader("Signal Selection")
        selected_signals = st.sidebar.multiselect("Choose signals to analyze", signals, default=signals)

        if len(selected_signals) >= 1:
            data_matrix = df[selected_signals].dropna().to_numpy()

            with st.spinner("Running anomaly detection..."):
                anomalies, scores, threshold = detect_anomalies(data_matrix, lookback=lookback, epochs=epochs)

            st.success(f"✅ Anomalies detected using threshold {threshold:.4f}")
            time_axis = np.arange(lookback, lookback + len(anomalies))

            for i, signal_name in enumerate(selected_signals):
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(time_axis, data_matrix[lookback:, i], label=signal_name, color='skyblue')
                anomaly_indices = np.where(anomalies)[0]
                ax.scatter(time_axis[anomaly_indices], data_matrix[lookback:, i][anomalies], color='red', label='Anomalies')
                ax.set_title(f"Anomaly Detection: {signal_name}")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Value")
                ax.legend()
                st.pyplot(fig)
        else:
            st.warning("Please select at least one signal column.")