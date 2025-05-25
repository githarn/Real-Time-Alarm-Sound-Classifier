import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer
import av

# Page configuration
st.set_page_config(page_title="🔔 Real-Time Alarm Classifier", layout="centered", page_icon="🔊")

# App title
st.markdown("<h1 style='text-align:center;'>🔊 Real-Time Alarm Sound Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a WAV file or use your mic to classify alarms and noises in real-time.</p>", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    - Upload a `.wav` file under 5 seconds **OR** use your microphone.
    - The model classifies the sound into:
      - 🔥 Fire alarm
      - 🛎️ Buzzer
      - 🚨 Smoke detector
      - ⏰ Timer alarm
      - 🚪 Opening door
      - 🐶 Barking
      - 💧 Water
      - 🚜 Lawn mower
    - Trained on demo data. Real-time classification may vary.
    """)

# Dummy model
CLASSES = ["Fire alarm", "Buzzer", "Smoke detector", "Timer alarm", "Opening door", "Barking", "Water", "Lawn mower"]

def dummy_train_model():
    X = np.random.rand(len(CLASSES)*20, 26)
    y = np.repeat(CLASSES, 20)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

model = dummy_train_model()

# Feature extraction
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(rms)])

# Tabs for Upload vs. Live
tab1, tab2 = st.tabs(["📂 Upload WAV File", "🎤 Live Microphone Input"])

# WAV Upload
with tab1:
    uploaded_file = st.file_uploader("Upload a WAV file (max ~5 sec)", type=["wav"])
    if uploaded_file:
        with st.spinner("🔎 Analyzing..."):
            y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
            features = extract_features(y, sr).reshape(1, -1)
            if features.shape[1] == model.n_features_in_:
                prediction = model.predict(features)[0]
                st.success(f"🎯 Prediction: **{prediction}**")
            else:
                st.error("⚠️ Could not extract correct feature size.")

# Live mic
with tab2:
    st.markdown("### 🔊 Start speaking to classify")
    st.caption("Ensure microphone permission is granted in your browser.")
    if "live_prediction" not in st.session_state:
        st.session_state["live_prediction"] = "Listening..."

    def audio_callback(frame: av.AudioFrame):
        audio = frame.to_ndarray(format="flt32")
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        sr = frame.sample_rate
        features = extract_features(audio, sr).reshape(1, -1)
        if features.shape[1] == model.n_features_in_:
            st.session_state["live_prediction"] = model.predict(features)[0]
        else:
            st.session_state["live_prediction"] = "⚠️ Feature mismatch"
        return frame

    webrtc_ctx = webrtc_streamer(
        key="audio-stream",
        audio_frame_callback=audio_callback,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    st.info(f"🔔 Detected: **{st.session_state['live_prediction']}**")

# Footer
st.markdown("---")
st.caption("🚧 Demo version — fine-tuning and training on real-world data will improve accuracy.")
