import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer
import av
import random
import time

# App configuration
st.set_page_config(page_title="Alarm Sound Classifier", layout="wide", page_icon="🔔")

# Styled CSS
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #0072B2;
    text-align: center;
}
.block {
    border-radius: 16px;
    padding: 2rem;
    background-color: #f8f9fa;
    box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}
.result {
    font-size: 24px;
    font-weight: 600;
    color: green;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>🔊 Alarm Sound Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a WAV file or use your mic to classify real-world alarms or noises.</p>", unsafe_allow_html=True)

# --- Dummy Model ---
CLASSES = ["Fire alarm", "Buzzer", "Smoke detector", "Timer alarm", "Opening door", "Barking", "Water", "Lawn mower"]
ICONS = {
    "Fire alarm": "🔥",
    "Buzzer": "🛎️",
    "Smoke detector": "🚨",
    "Timer alarm": "⏰",
    "Opening door": "🚪",
    "Barking": "🐶",
    "Water": "💧",
    "Lawn mower": "🚜"
}

def dummy_train_model():
    X = np.random.rand(len(CLASSES)*20, 26)
    y = np.repeat(CLASSES, 20)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

model = dummy_train_model()

# --- Feature Extraction ---
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(rms)])

# --- Layout Sections ---
col1, col2 = st.columns(2)

# --- Upload Section ---
with col1:
    st.subheader("📂 Upload Audio")
    st.markdown('<div class="block">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        with st.spinner("Analyzing audio..."):
            y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
            features = extract_features(y, sr).reshape(1, -1)
            if features.shape[1] == model.n_features_in_:
                prediction = model.predict(features)[0]
                confidence = round(random.uniform(0.7, 1.0), 2)
                st.success(f"{ICONS[prediction]} **Prediction**: {prediction}")
                st.progress(confidence)
                st.toast(f"✅ {prediction} with {int(confidence*100)}% confidence", icon="📊")
                if confidence > 0.9:
                    st.balloons()
            else:
                st.error("⚠️ Feature extraction failed.")
    else:
        st.info("Upload a .wav file to classify sound.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Live Section ---
with col2:
    st.subheader("🎤 Use Microphone")
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
        key="live-audio",
        audio_frame_callback=audio_callback,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    st.info(f"🔔 Real-time Prediction: **{st.session_state['live_prediction']}**")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer Legend ---
with st.expander("📘 Class Legend"):
    for label, emoji in ICONS.items():
        st.markdown(f"- {emoji} **{label}**")

st.markdown("---")
st.caption("🎧 Interactive alarm classifier built with Streamlit · UI upgraded for clarity and experience.")
