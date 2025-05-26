import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer
import av

# -------------------- Page Config --------------------
st.set_page_config(page_title="Real-Time Alarm Sound Classifier", layout="wide", page_icon="🔔")

# -------------------- Custom CSS --------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #f0f9ff, #cbebff);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }

    .stApp {
        background: linear-gradient(135deg, #d4f1f9 0%, #ffffff 100%);
    }

    .stButton>button {
        background-color: #0099cc;
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #007399;
    }

    .stFileUploader {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #0072B2;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    .custom-card {
        padding: 1.5rem;
        background: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }

    .title-text {
        font-size: 2rem;
        font-weight: bold;
        color: #0072B2;
    }

    .subtitle-text {
        font-size: 1.2rem;
        margin-top: -10px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- Sidebar Info --------------------
with st.sidebar:
    st.markdown("## ℹ️ About this App")
    st.write(
        """
        This app classifies common alarm and noise sounds such as:

        **ALARM**
        - 🔥 Fire alarm         
        - 🛎️ Buzzer             
        - 🚨 Smoke detector     
        - ⏰ Timer alarm 

        **NOISE**
        - 🚪 Opening door       
        - 🐶 Barking            
        - 💧 Water              
        - 🚜 Lawn mower         

        Upload a WAV file or use your microphone for live classification.

        *Note*: Model is trained on sample data for demo purposes.
        """
    )

# -------------------- Page Title --------------------
st.markdown('<h1 class="title-text">🔊 Real-Time Alarm Sound Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload a WAV file or use your microphone to classify the sound.</p>', unsafe_allow_html=True)

# -------------------- Classes --------------------
CLASSES = [
    "Fire alarm", "Buzzer", "Smoke detector", "Timer alarm",
    "Opening door", "Barking", "Water", "Lawn mower"
]

# -------------------- Dummy Model --------------------
def dummy_train_model():
    feature_len = 26
    X = np.random.rand(len(CLASSES)*20, feature_len)
    y = np.repeat(CLASSES, 20)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

model = dummy_train_model()

# -------------------- Feature Extraction --------------------
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    rms_mean = np.mean(rms)
    return np.hstack([mfccs_mean, chroma_mean, rms_mean])

# -------------------- Upload & Prediction Columns --------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("📂 Upload WAV file")
    uploaded_file = st.file_uploader("Select a WAV file to classify", type=['wav'], key="upload")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    if uploaded_file:
        with st.spinner("Analyzing uploaded audio..."):
            y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
            features = extract_features(y, sr).reshape(1, -1)
            if features.shape[1] == model.n_features_in_:
                prediction = model.predict(features)[0]
                st.success(f"🎯 **Prediction:** {prediction}")
            else:
                st.error("⚠️ Feature size mismatch. Could not classify.")
    else:
        st.info("📁 Upload a file to see prediction here.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Live Microphone Classification --------------------
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.header("🎤 Live microphone classification")

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
        st.session_state["live_prediction"] = "⚠️ Feature size mismatch"
    return frame

webrtc_ctx = webrtc_streamer(
    key="live-alarm-classifier",
    audio_frame_callback=audio_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if "live_prediction" in st.session_state:
    st.success(f"🔊 Live prediction: **{st.session_state['live_prediction']}**")
else:
    st.info("🎙️ Waiting for live audio input...")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown(
    """
    <hr>
    <center>
    <small>🔔 Real-Time Sound Classifier | Demo Version | Built with ❤️ using Streamlit</small>
    </center>
    """,
    unsafe_allow_html=True
)
