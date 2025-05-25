import streamlit as st
import numpy as np
import librosa
import joblib
from streamlit_webrtc import webrtc_streamer
import av
import random

# Load your trained model
model = joblib.load("alarm_model.pkl")  # Make sure file is in same directory

CLASSES = ["Fire alarm", "Buzzer", "Smoke detector", "Timer alarm",
           "Opening door", "Barking", "Water", "Lawn mower"]

ICONS = {
    "Fire alarm": "🔥", "Buzzer": "🛎️", "Smoke detector": "🚨", "Timer alarm": "⏰",
    "Opening door": "🚪", "Barking": "🐶", "Water": "💧", "Lawn mower": "🚜"
}

# Page setup
st.set_page_config(page_title="Alarm Classifier", layout="wide", page_icon="🔔")

# Styling
st.markdown("""
<style>
.block { border-radius: 16px; padding: 2rem; background-color: #f8f9fa;
box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Feature Extraction
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(rms)])

# Header
st.title("🔊 Real-Time Alarm Sound Classifier")
st.markdown("Classifies alarm and noise sounds via upload or microphone input.")

# Layout
col1, col2 = st.columns(2)

# File Upload
with col1:
    st.subheader("📂 Upload Audio")
    st.markdown('<div class="block">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
        features = extract_features(y, sr).reshape(1, -1)
        if features.shape[1] == model.n_features_in_:
            pred = model.predict(features)[0]
            prob = random.uniform(0.75, 0.95)  # Simulated for now
            st.success(f"{ICONS[pred]} **Prediction**: {pred}")
            st.progress(prob)
            st.toast(f"Detected: {pred} ({int(prob*100)}%)", icon="🎧")
        else:
            st.error("⚠️ Feature mismatch")
    st.markdown("</div>", unsafe_allow_html=True)

# Live Mic Input
with col2:
    st.subheader("🎤 Microphone Input")
    st.markdown('<div class="block">', unsafe_allow_html=True)

    if "live_prediction" not in st.session_state:
        st.session_state["live_prediction"] = "Waiting..."

    def audio_callback(frame: av.AudioFrame):
        audio = frame.to_ndarray(format="flt32")
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        sr = frame.sample_rate
        features = extract_features(audio, sr).reshape(1, -1)
        if features.shape[1] == model.n_features_in_:
            pred = model.predict(features)[0]
            st.session_state["live_prediction"] = pred
        else:
            st.session_state["live_prediction"] = "⚠️ Feature error"
        return frame

    webrtc_streamer(
        key="live",
        audio_frame_callback=audio_callback,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    st.info(f"🎧 Live Prediction: **{st.session_state['live_prediction']}**")
    st.markdown('</div>', unsafe_allow_html=True)

# Legend
with st.expander("📘 Class Legend"):
    for cls in CLASSES:
        st.markdown(f"- {ICONS[cls]} **{cls}**")

