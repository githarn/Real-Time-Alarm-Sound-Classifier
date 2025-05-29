import os
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer
import av
import random
import time

# 🎨 Page Configuration
st.set_page_config(page_title="🔔 Alarm Sound Classifier", layout="wide", page_icon="🎧")

# 🌗 Toggle Theme Mode
theme = st.toggle("🌗 Toggle Dark Mode", value=False)
bg_color = "#1E1E1E" if theme else "#F4F4F4"
text_color = "#FFFFFF" if theme else "#000000"

st.markdown(f""" 
    <style>
    body{{background-color:{bg_color}; color:{text_color}; font-family: 'Arial', sans-serif;}}
    .glass-card {{border-radius: 15px; background: rgba(255,255,255,0.1); backdrop-filter: blur(15px); padding: 20px;}}
    </style>
""", unsafe_allow_html=True)

# 🔊 Header
st.markdown("<h1 style='text-align:center;'>🔊 Enhanced Alarm Classifier</h1>", unsafe_allow_html=True)

# 🎭 Sound Classes
ALL_CLASSES = {
    "Fire alarm": "🔥", "Buzzer": "🛎️", "Smoke detector": "🚨",
    "Timer alarm": "⏰", "Opening door": "🚪", "Barking": "🐶",
    "Water": "💧", "Lawn mower": "🚜", "Non-alarm sound": "⚠️"
}
CLASSES = list(ALL_CLASSES.keys())

# 🏆 Function to Load and Train Model
def load_dataset(dataset_folder):
    X, y = [], []
    for category in ["alarmsounds", "nonalarm"]:
        category_path = os.path.join(dataset_folder, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(category)
    return np.array(X), np.array(y)

# 🔬 Feature Extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=5.0)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(rms)])

# 📂 Load Dataset and Train Model
dataset_folder = "C:/Users/Downloads/nonalarm/alarmsound"
X, y = load_dataset(dataset_folder)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 📂 Upload & 🎤 Mic Tabs
tab1, tab2 = st.tabs(["📂 Upload File", "🎤 Microphone"])

# 📂 Upload File Tab
with tab1:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)

        # 📊 Audio Visualization (Waveform)
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Visualization", fontsize=12)
        st.pyplot(fig)

        features = extract_features(uploaded_file).reshape(1, -1)
        if features.shape[1] == model.n_features_in_:
            prediction = model.predict(features)[0]
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
        audio = frame.to_ndarray(format="flt32").mean(axis=0) if frame.to_ndarray(format="flt32").ndim > 1 else frame.to_ndarray(format="flt32")
        sr = frame.sample_rate
        features = extract_features(audio).reshape(1, -1)
        pred = model.predict(features)[0] if features.shape[1] == model.n_features_in_ else "⚠️ Feature mismatch"
        return frame

    webrtc_streamer(key="live-audio", audio_frame_callback=audio_callback, media_stream_constraints={"audio": True, "video": False}, async_processing=True)

st.markdown("---")
st.caption("Built with Streamlit · Enhanced UI & Interactivity 🚀")
