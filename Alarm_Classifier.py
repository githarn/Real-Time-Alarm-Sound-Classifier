import streamlit as st
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("🔊 Real-Time Alarm Sound Classifier ")

uploaded_files = st.file_uploader("Upload WAV files", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    st.write("Enter label for each uploaded audio file:")
    labels = []
    for file in uploaded_files:
        label = st.text_input(f"Label for {file.name}", key=file.name)
        labels.append(label.strip())

    if all(labels) and len(labels) == len(uploaded_files):
        
        @st.cache_data
        def extract_features_safe(file):
            try:
                y, sr = librosa.load(file, duration=5.0, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)

                mfccs_mean = np.mean(mfccs, axis=1)
                chroma_mean = np.mean(chroma, axis=1)
                rms_mean = np.mean(rms)

                return np.hstack([mfccs_mean, chroma_mean, rms_mean])
            except Exception as e:
                st.warning(f"Error processing {file.name}: {e}")
                return None

        features = []
        valid_labels = []

        st.info("Extracting features from uploaded audio files...")
        for file, label in zip(uploaded_files, labels):
            feats = extract_features_safe(file)
            if feats is not None:
                features.append(feats)
                valid_labels.append(label)

        if len(features) < 2:
            st.error("❌ Need at least 2 valid audio files with labels to train and test the model.")
        else:
            X = np.array(features)
            y = np.array(valid_labels)

            unique_classes = set(y)
            if len(unique_classes) < 2:
                st.error("❌ Need at least 2 different classes/labels to train a classifier.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if len(X_train) < 3:
                    st.error("❌ Not enough training samples for KNN with n_neighbors=3. Upload more labeled audio files.")
                else:
                    model = KNeighborsClassifier(n_neighbors=3)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Extracted features from {len(features)} audio files.")
                    st.metric("🎯 Model Accuracy", f"{acc * 100:.2f}%")

    else:
        st.info("Please enter a label for each uploaded audio file.")
else:
    st.info("Please upload WAV files to start.")
