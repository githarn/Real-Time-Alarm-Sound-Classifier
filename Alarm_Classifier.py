import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier

st.title("🔊 Single WAV Alarm Sound Classifier")

# --- Example training data for demonstration ---
# Replace this with your own training files and labels if you want real training
@st.cache_data(show_spinner=False)
def example_train_model():
    # Dummy features for "fire alarm" and "any alarm" sounds
    # Just random for demo purposes — replace with real features!
    X = np.array([
        np.random.rand(29),  # features length from MFCC+Chroma+RMS
        np.random.rand(29),
        np.random.rand(29),
        np.random.rand(29),
        np.random.rand(29)
    ])
    y = np.array(['fire alarm', 'fire alarm', 'any alarm', 'any alarm', 'any alarm'])
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)
    return model

model = example_train_model()

def extract_features(file):
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
        st.error(f"Error processing file: {e}")
        return None

uploaded_file = st.file_uploader("Upload a WAV file to classify", type=["wav"])

if uploaded_file:
    features = extract_features(uploaded_file)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        st.success(f"Predicted alarm sound: **{prediction.capitalize()}**")
