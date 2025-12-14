# ================================
# EEG Neuromarketing Prediction App
# Academic Version (Single EEG Inference + Optional Evaluation)
# Binary Classification: Like vs Dislike
# ================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ================================
# Load Model & Scaler
# ================================
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model("model.h5", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ================================
# Helper Functions
# ================================

def load_signal_txt(file):
    return np.loadtxt(file)


def load_lab_file(file):
    labels = []
    for line in file:
        clean = line.decode("utf-8").strip().lower()
        if clean == "like":
            labels.append(1)
        elif clean == "disike":
            labels.append(0)
        elif clean == "neutral":
            labels.append(2)
    return np.array(labels)


# Bandpass filter EEG 1–50 Hz

def bandpass_filter(signal, lowcut=1.0, highcut=50.0, fs=250, order=5):
    if signal.shape[0] < 33:
        return signal
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


# ================================
# Streamlit Layout
# ================================

st.title("EEG Neuromarketing Prediction")

st.markdown("""
Aplikasi ini mendukung dua skenario:
1. **Prediksi tunggal**: upload **1 file EEG (.txt)** → output **Like / Dislike**
2. **Evaluasi model (opsional)**: upload file **.lab** untuk melihat performa model

Model menggunakan **binary classification (Like vs Dislike)**.
Label **Neutral tidak disertakan dalam evaluasi**.
""")

signal_file = st.file_uploader("Upload EEG file (.txt)", type=["txt"], accept_multiple_files=False)
label_file = st.file_uploader("(Opsional) Upload Label file (.lab)", type=["lab"], accept_multiple_files=False)

# ================================
# Single EEG Prediction
# ================================

if signal_file:
    X = load_signal_txt(signal_file)
    X = bandpass_filter(X)

    st.subheader("EEG Signal Preview")
    st.line_chart(X[:500, 0])

    # Preprocessing
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Prediction
    preds = model.predict(X_scaled)
    preds_class = (preds >= 0.5).astype(int).flatten()

    # Majority voting
    final_pred = int(np.round(np.mean(preds_class)))
    final_label = "Like" if final_pred == 1 else "Disike"

    st.subheader("Prediction Result")
    st.success(f"Predicted Response: **{final_label}**")

    # ================================
    # Optional Evaluation
    # ================================

    if label_file:
        y = load_lab_file(label_file.readlines())
        min_len = min(len(y), len(preds_class))
        y = y[:min_len]
        preds_eval = preds_class[:min_len]

        # Remove neutral
        mask = y != 2
        y = y[mask]
        preds_eval = preds_eval[mask]

        if len(y) == 0:
            st.warning("Label hanya berisi Neutral, evaluasi tidak dapat dilakukan.")
        else:
            st.subheader("Evaluation Results")
            accuracy = np.mean(y == preds_eval)
            st.success(f"Accuracy: {accuracy * 100:.2f}%")

            report = classification_report(
                y,
                preds_eval,
                labels=[0, 1],
                target_names=['Disike', 'Like'],
                digits=4
            )
            st.text(report)

            cm = confusion_matrix(y, preds_eval, labels=[0, 1])
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=['Disike', 'Like']).plot(ax=ax, cmap=plt.cm.Blues)
            st.pyplot(fig)
