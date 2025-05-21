import streamlit as st
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("🔊 Real-Time Alarm Sound Classifier (Offline Model Training)")

# === Upload Files ===
uploaded_csv = st.file_uploader("Upload Metadata CSV", type=["csv"])
audio_dir = st.text_input("Enter path to folder containing WAV files")

if uploaded_csv and audio_dir:
    metadata = pd.read_csv(uploaded_csv)

    @st.cache_data
    def extract_features_safe(file_path):
        try:
            y, sr = librosa.load(file_path, duration=5.0)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)

            mfccs_mean = np.mean(mfccs, axis=1)
            chroma_mean = np.mean(chroma, axis=1)
            rms_mean = np.mean(rms)

            return np.hstack([mfccs_mean, chroma_mean, rms_mean])
        except Exception as e:
            st.warning(f"Error processing {file_path}: {e}")
            return None

    st.info("Extracting features from audio files...")
    features, labels = [], []

    for _, row in metadata.iterrows():
        file_name = row['file_name']
        label = row['label']
        file_path = os.path.join(audio_dir, file_name)
        feats = extract_features_safe(file_path)
        if feats is not None:
            features.append(feats)
            labels.append(label)

    if features:
        st.success(f"✅ Extracted features from {len(features)} audio files.")

        # Train classifier
        X = np.array(features)
        y = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("🎯 Model Accuracy", f"{acc * 100:.2f}%")
    else:
        st.error("❌ No valid audio features extracted.")

