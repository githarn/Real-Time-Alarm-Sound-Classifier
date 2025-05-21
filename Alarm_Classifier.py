import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer
import av

# Page config for better UX
st.set_page_config(page_title="Real-Time Alarm Sound Classifier", layout="wide", page_icon="🔔")

# Inject background gradient with CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #d4f1f9 0%, #0b486b 100%);
        color: #222;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #0072B2;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #005b8f;
        cursor: pointer;
    }
    .stFileUploader>div>div>input {
        border-radius: 8px;
        border: 2px solid #0072B2;
        padding: 12px;
    }
    .reportview-container .markdown-text-container {
        font-size: 20px;
    }
    .css-1d391kg {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with info and instructions
with st.sidebar:
    st.markdown("## ℹ️ About this App")
    st.write(
        """
        This app classifies common alarm and noise sounds such as:
        - Fire alarm
        - Buzzer
        - Smoke detector
        - Timer alarm
        - Opening door
        - Barking
        - Water
        - Lawn mower
        
        Upload a WAV file or use your microphone for live classification.
        
        *Note*: Model is trained on sample data for demo purposes.
        """
    )
    
# Title and description
st.title("🔊 Real-Time Alarm Sound Classifier 🔔")
st.markdown(
    """
    Upload a WAV file or speak into your microphone to classify the sound.
    """
)

# Define classes
CLASSES = [
    "Fire alarm", "Buzzer", "Smoke detector", "Timer alarm",
    "Opening door", "Barking", "Water", "Lawn mower"
]

# Dummy model training
def dummy_train_model():
    feature_len = 26
    X = np.random.rand(len(CLASSES)*20, feature_len)  # More samples per class for demo
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
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    rms_mean = np.mean(rms)
    return np.hstack([mfccs_mean, chroma_mean, rms_mean])

# Use columns for upload + results side by side
col1, col2 = st.columns([2, 3])

with col1:
    st.header("📂 Upload WAV file")
    uploaded_file = st.file_uploader("Select a WAV file to classify", type=['wav'], key="upload")

with col2:
    if uploaded_file:
        with st.spinner("Analyzing uploaded audio..."):
            y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
            features = extract_features(y, sr).reshape(1, -1)
            if features.shape[1] == model.n_features_in_:
                prediction = model.predict(features)[0]
                st.success(f"🎯 **Prediction:** {prediction}")
            else:
                st.error("⚠️ Feature size mismatch. Could not classify.")

st.markdown("---")

# Live microphone classification
st.header("🎤 Live microphone classification")

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
    st.info("Waiting for live audio input...")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:12px; color:#fff;">
    Built with ❤️ using Streamlit and Librosa
    </p>
    """,
    unsafe_allow_html=True,
)
