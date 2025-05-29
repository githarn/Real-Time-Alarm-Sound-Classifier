import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer
import av
import random
import time

# 🎨 Page Configuration
st.set_page_config(page_title="🔔 Real-Time Alarm Classifier", layout="centered", page_icon="🎧")

# 🌗 Toggle Dark Mode
theme = st.toggle("🌗 Toggle Dark Mode", value=False)
bg_color = "#1E1E1E" if theme else "#FFFFFF"
text_color = "#FFFFFF" if theme else "#000000"

# 🔧 Apply Style
st.markdown(
    f"""
    <style>
    .dark-container {{
        background-color: {bg_color};
        color: {text_color};
        padding: 30px;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #0072B2;
        color: white;
    }}
    </style>
    <div class="dark-container">
    """,
    unsafe_allow_html=True,
)
# 🔊 Header
st.markdown(f"<h1 style='text-align:center;'>🔊 Real-Time Alarm Sound Classifier</h1>", unsafe_allow_html=True)

# 🧠 Help Section
with st.expander("🧠 How does this work?"):
    st.markdown("""
    - Upload a short `.wav` file or use your mic.
    - The system classifies sounds into:
      - 🔥 Fire alarm, 🛎️ Buzzer, 🚨 Smoke detector, ⏰ Timer alarm  
      - 🚪 Opening door, 🐶 Barking, 💧 Water, 🚜 Lawn mower
    - A simulated confidence bar shows how confident the model is.
    - Results update in real-time for microphone input.
    """)

# 🎭 Class Definitions
ALL_CLASSES = {
    "Fire alarm": "🔥",
    "Buzzer": "🛎️",
    "Smoke detector": "🚨",
    "Timer alarm": "⏰",
    "Opening door": "🚪",
    "Barking": "🐶",
    "Water": "💧",
    "Lawn mower": "🚜"
}

# 🎚️ Filter Option
sound_type = st.radio("🎚️ Filter Sound Types", ["All", "Alarm", "Noise"], horizontal=True)
if sound_type == "Alarm":
    CLASSES = ["Fire alarm", "Buzzer", "Smoke detector", "Timer alarm"]
elif sound_type == "Noise":
    CLASSES = ["Opening door", "Barking", "Water", "Lawn mower"]
else:
    CLASSES = list(ALL_CLASSES.keys())

# 🧠 Dummy Model for Demo
def dummy_train_model():
    X = np.random.rand(len(CLASSES)*20, 26)
    y = np.repeat(CLASSES, 20)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

model = dummy_train_model()

# 🧪 Feature Extraction
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(rms)])

# 📂 Tabs
tab1, tab2 = st.tabs(["📂 Upload File", "🎤 Microphone"])

# 📂 Upload Tab
with tab1:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        with st.spinner("Analyzing..."):
            y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
            features = extract_features(y, sr).reshape(1, -1)
            if features.shape[1] == model.n_features_in_:
                prediction = model.predict(features)[0]
                confidence = random.uniform(0.75, 1.0)  # Simulated
                st.success(f"{ALL_CLASSES[prediction]} **{prediction}** detected!")
                st.progress(confidence)
                st.toast(f"✅ Classifier is {int(confidence * 100)}% confident.", icon="🤖")
            else:
                st.error("⚠️ Feature size mismatch.")

# 🎤 Live Mic Tab
with tab2:
    st.markdown("### 🎙️ Speak into your mic")
    st.caption("Allow mic permissions in your browser.")

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
            st.session_state["live_prediction"] = "⚠️ Feature mismatch"
        return frame

    webrtc_streamer(
        key="live-audio",
        audio_frame_callback=audio_callback,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    st.info(f"🔔 Real-time: **{ALL_CLASSES.get(st.session_state['live_prediction'], '')} {st.session_state['live_prediction']}**")

# 📘 Sound Class Legend
with st.expander("📘 Sound Labels Legend"):
    for name, emoji in ALL_CLASSES.items():
        st.markdown(f"- {emoji} **{name}**")

# 📍 Footer
st.markdown("---")
st.caption("🔧 Built with Streamlit · Demo classifier · UI enhanced for interactivity.")

# 🔚 Close the container div
st.markdown("</div>", unsafe_allow_html=True)
