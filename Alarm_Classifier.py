import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier

st.title("🔊 Alarm Sound Classifier: Fire, Tsunami, Timer")

# For demo, create dummy training data with 3 classes
@st.cache_data(show_spinner=False)
def example_train_model():
    feature_length = 29  # must match features extracted below

    # Generate random dummy data for each alarm type (replace with real training data)
    X = np.random.rand(9, feature_length)  # 3 samples per class
    y = np.array([
        'fire alarm',
        'fire alarm',
        'fire alarm',
        'tsunami alarm',
        'tsunami alarm',
        'tsunami alarm',
        'timer alarm',
        'timer alarm',
        'timer alarm',
    ])
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

        features = np.hstack([mfccs_mean, chroma_mean, rms_mean])
        st.write(f"Extracted features shape: {features.shape}")
        return features
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
