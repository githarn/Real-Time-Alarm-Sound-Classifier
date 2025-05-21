import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

st.title("🔊 Alarm Sound Classifier with Real Training")

uploaded_files = st.file_uploader("Upload multiple labeled WAV files (zip or individual)", accept_multiple_files=True, type=['wav'])
labels_input = st.text_input("Enter corresponding labels separated by commas (e.g. fire,fire,timer)")

def extract_features(file):
    y, sr = librosa.load(file, duration=5.0, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    rms_mean = np.mean(rms)
    return np.hstack([mfccs_mean, chroma_mean, rms_mean])

if uploaded_files and labels_input:
    labels = [lbl.strip() for lbl in labels_input.split(',')]
    if len(labels) != len(uploaded_files):
        st.error("Number of labels does not match number of files!")
    else:
        features = []
        for file in uploaded_files:
            f = extract_features(file)
            features.append(f)
        X = np.array(features)
        y = np.array(labels)

        # Train/test split just to show accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained with accuracy: {acc*100:.2f}%")

        st.markdown("---")
        st.subheader("Upload one WAV file to classify:")

        test_file = st.file_uploader("Upload WAV to classify", type=['wav'])
        if test_file:
            test_feat = extract_features(test_file).reshape(1, -1)
            prediction = model.predict(test_feat)[0]
            st.write(f"Predicted alarm type: **{prediction}**")
