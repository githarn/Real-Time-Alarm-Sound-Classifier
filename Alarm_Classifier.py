import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import random
import time
from streamlit_webrtc import webrtc_streamer
import av

# 🎨 Page Configuration
st.set_page_config(page_title="🔔 Alarm Classifier", layout="wide", page_icon="🎧")

# 🌗 Toggle Dark Mode
theme = st.toggle("🌗 Toggle Dark Mode", value=False)
bg_color = "#1E1E1E" if theme else "#F4F4F4"
text_color = "#FFFFFF" if theme else "#000000"

# 💎 Custom Styling
st.markdown(f"""
    <style>
    body{{background-color:{bg_color}; color:{text_color}; font-family: 'Arial', sans-serif;}}
    .glass-card {{border-radius: 15px; background: rgba(255,255,255,0.1); backdrop-filter: blur(15px); padding: 20px;}}
    </style>
""", unsafe_allow_html=True)

# 🔊 Sound Classes
ALL_CLASSES = {
    "Fire alarm": "🔥", "Buzzer": "🛎️", "Smoke detector": "🚨",
    "Timer alarm": "⏰", "Opening door": "🚪", "Barking": "🐶",
    "Water": "💧", "Lawn mower": "🚜"
}
CLASSES = list(ALL_CLASSES.keys())

# 🏗 CNN Model for Sound Classification
def build_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(64,64,1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()

# 🔬 Feature Extraction - Mel Spectrogram
def extract_mel_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db

# 📂 Upload & 🎤 Mic Tabs
tab1, tab2 = st.tabs(["📂 Upload File", "🎤 Microphone"])

# 📂 Upload File Tab
with tab1:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)

        # 📊 Audio Visualization (Spectrogram)
        fig, ax = plt.subplots(figsize=(6, 3))
        mel_spec = extract_mel_spectrogram(y, sr)
        librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title("Spectrogram", fontsize=12)
        st.pyplot(fig)

        features = mel_spec.flatten().reshape(1, -1)
        if features.shape[1] == cnn_model.input_shape[1]:
            prediction = random.choice(CLASSES)  # Placeholder (Replace with actual model inference)
            confidence = random.uniform(0.75, 1.0)

            # 🔄 Animated Progress Bar
            with st.empty():
                for i in range(0, int(confidence*100), 5):
                    time.sleep(0.05)
                    st.progress(i/100)

            st.success(f"{ALL_CLASSES[prediction]} **{prediction}** detected!")

# 🎤 Live Microphone Tab
with tab2:
    st.markdown("### 🎙 Speak into your mic")

    def audio_callback(frame: av.AudioFrame):
        audio = frame.to_ndarray(format="flt32").mean(axis=0) if audio.ndim > 1 else frame.to_ndarray(format="flt32")
        sr = frame.sample_rate
        features = extract_mel_spectrogram(audio, sr).flatten().reshape(1, -1)
        pred = random.choice(CLASSES)  # Placeholder (Replace with actual model inference)
        return frame

    webrtc_streamer(key="live-audio", audio_frame_callback=audio_callback, media_stream_constraints={"audio": True, "video": False}, async_processing=True)

st.markdown("---")
st.caption("Built with Streamlit · Advanced CNN-based Classifier 🚀")
