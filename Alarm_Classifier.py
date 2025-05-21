import streamlit as st
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("🔊 Real-Time Alarm Sound Classifier (Offline Model Training)")

# Upload multiple wav files
uploaded_files = st.file_uploader("Upload WAV files", type=["wav"], accept_multiple_files=True)

# Let user enter the label for all uploaded files (or you can extend to per-file labeling)
label = st.text_input("Enter label for all uploaded audio files")

if uploaded_files and label:
    @st.cache_data
    def extract_features_safe(file):
        try:
            # librosa accepts file-like objects, but sometimes better to read bytes
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

    st.info("Extracting features from uploaded audio files...")

    features = []
    labels = []

    for file in uploaded_files:
        feats = extract_features_safe(file)
        if feats is not None:
            features.append(feats)
            labels.append(label)  # same label for all files

    if features:
        st.success(f"✅ Extracted features from {len(features)} audio files.")

        X = np.array(features)
        y = np.array(labels)

        # For demonstration, we split the data even if all have same label
        # In practice, you'd want multiple labels and files per label
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("🎯 Model Accuracy", f"{acc * 100:.2f}%")
    else:
        st.error("❌ No valid audio features extracted.")
else:
    st.info("Please upload WAV files and enter a label to start training.")
