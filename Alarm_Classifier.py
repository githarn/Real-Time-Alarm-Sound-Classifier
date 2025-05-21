import streamlit as st
import numpy as np
import librosa
import joblib
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title="Alarm & Noise Sound Classifier", layout="centered")
st.title("🔊 Alarm & Noise Sound Classifier")

# Load pre-trained model
model = joblib.load("alarm_sound_model.pkl")  # Make sure this file is in the same directory or provide correct path

# Define expected feature length
EXPECTED_FEATURE_LEN = model.n_features_in_

# Define classes (for display only)
CLASSES = model.classes_ if hasattr(model, 'classes_') else []

# Feature extraction function
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    rms_mean = np.mean(rms)
    return np.hstack([mfccs_mean, chroma_mean, rms_mean])

# WAV file classification
st.header("Upload a WAV file to classify")
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file is not None:
    try:
        y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
        features = extract_features(y, sr).reshape(1, -1)

        if features.shape[1] == EXPECTED_FEATURE_LEN:
            prediction = model.predict(features)[0]
            st.success(f"Predicted sound: **{prediction}**")
        else:
            st.error("Feature size mismatch. Model expects a different number of features.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Live audio classification
st.markdown("---")
st.header("Or classify live audio from your microphone")

def audio_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray(format="flt32")
    if audio.ndim > 1:
        audio = audio.mean(axis=0)  # Convert stereo to mono
    sr = frame.sample_rate

    features = extract_features(audio, sr).reshape(1, -1)
    if features.shape[1] == EXPECTED_FEATURE_LEN:
        pred = model.predict(features)[0]
        st.session_state["live_prediction"] = pred
    else:
        st.session_state["live_prediction"] = "Feature size mismatch"

    return frame

webrtc_ctx = webrtc_streamer(
    key="live-alarm-classifier",
    audio_frame_callback=audio_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if "live_prediction" in st.session_state:
    st.success(f"Live audio prediction: **{st.session_state['live_prediction']}**")
else:
    st.info("Waiting for live audio input...")
