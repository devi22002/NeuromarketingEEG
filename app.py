# app_multi_final.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Load Model & Scaler
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# Functions
def load_signal_txt(file):
    data = np.loadtxt(file)
    return data

def load_lab_file(file):
    labels = []
    for line in file:
        clean = line.decode("utf-8").strip()
        if clean == "":
            continue
        if clean.lower() == "like":
            labels.append(1)
        elif clean.lower() == "disike":
            labels.append(0)
        elif clean.lower() == "neutral":
            labels.append(2)
    return np.array(labels)

# Bandpass filter EEG 1-50 Hz
def bandpass_filter(signal, lowcut=1.0, highcut=50.0, fs=250, order=5):
    # filtfilt padlen = 3*(max(len(a), len(b)) -1) = ~33 for order=5
    if signal.shape[0] < 33:
        return signal  # skip filter untuk sinyal terlalu pendek
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, axis=0)
    return filtered

# Streamlit Layout
st.title("EEG Neuromarketing Prediction (Multi-File)")

st.write("""
Upload beberapa file EEG (.txt) dan label (.lab) sekaligus untuk prediksi.
Pastikan nama file EEG dan label sama sebelum ekstensi.
""")

signal_files = st.file_uploader("Upload EEG files (.txt)", type=["txt"], accept_multiple_files=True)
label_files = st.file_uploader("Upload Label files (.lab)", type=["lab"], accept_multiple_files=True)

if signal_files and label_files:
    X_all = []
    y_all = []
    skipped_files = []

    # Buat dictionary label untuk matching
    label_dict = {file.name.replace(".lab",""): file for file in label_files}

    for sfile in signal_files:
        fname = sfile.name.replace(".txt","")
        if fname not in label_dict:
            skipped_files.append(fname)
            continue

        # Load signal & label
        X = load_signal_txt(sfile)
        y = load_lab_file(label_dict[fname].readlines())

        # Sesuaikan panjang jika mismatch
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        # Filter EEG
        X = bandpass_filter(X)

        X_all.append(X)
        y_all.append(y)

    if skipped_files:
        st.warning(f"File tanpa label akan dilewati: {', '.join(skipped_files)}")

    # Gabungkan semua data
    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    # EDA sederhana
    n_samples, n_channels = X_all.shape
    n_labels = len(np.unique(y_all))
    st.write(f"Total sampel: {n_samples}, Jumlah channel: {n_channels}, Jumlah label unik: {n_labels}")

    st.line_chart(X_all[:500,0])  # contoh plot channel pertama 500 sampel

    # Preprocessing
    X_scaled = scaler.transform(X_all)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Prediction
    preds = model.predict(X_scaled)
    preds_class = (preds > 0.5).astype(int).flatten()

    # Show sample prediction
    st.write("Predicted labels (sample 20):", preds_class[:20])
    comparison_df = pd.DataFrame({"Actual": y_all[:len(preds_class)], "Predicted": preds_class})
    st.dataframe(comparison_df)

    # Evaluation
    accuracy = np.mean(preds_class == y_all[:len(preds_class)])
    st.success(f"Prediction Accuracy: {accuracy*100:.2f}%")

    # Pastikan labels sesuai
    unique_labels = np.unique(y_all[:len(preds_class)])
    label_names = []
    if 0 in unique_labels:
        label_names.append('Dislike')
    if 1 in unique_labels:
        label_names.append('Like')

    # Classification report
    report = classification_report(
        y_all[:len(preds_class)],
        preds_class,
        labels=unique_labels,
        target_names=label_names
    )
    st.text(report)

    # Confusion matrix
    cm = confusion_matrix(y_all[:len(preds_class)], preds_class, labels=unique_labels)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    st.pyplot(fig)