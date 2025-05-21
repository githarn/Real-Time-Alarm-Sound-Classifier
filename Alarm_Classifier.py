import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer
import av

st.title("🔊 Alarm & Noise Sound Classifier")

# Define classes
CLASSES = [
    "Fire alarm", "Buzzer", "Smoke detector", "Timer alarm",
    "Opening door", "Barking", "Water", "Lawn mower"
]

# Dummy training data for demonstration
def dummy_train_model():
    feature_len = 26  # 13 mfcc + 12 chroma + 1 rms = 26, plus maybe padding
    X = np.random.rand(len(CLASSES)*5, feature_len)  # 5 samples per class random data
    y = np.repeat(CLASSES, 5)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

model = dummy_train_model()

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    rms_mean = np.mean(rms)
    return np.hstack([mfccs_mean, chroma_mean, rms_mean])

st.header("Upload a WAV file to classify")
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
    features = extract_features(y, sr).reshape(1, -1)

    if features.shape[1] == model.n_features_in_:
        prediction = model.predict(features)[0]
        st.success(f"Predicted sound: **{prediction}**")
    else:
        st.error("Feature size mismatch.")

st.markdown("---")
st.header("Or classify live audio from your microphone")

def audio_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray(format="flt32")
    if audio.ndim > 1:
        audio = audio.mean(axis=0)  # stereo to mono
    sr = frame.sample_rate

    features = extract_features(audio, sr).reshape(1, -1)
    if features.shape[1] == model.n_features_in_:
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
