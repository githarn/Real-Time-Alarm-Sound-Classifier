import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer
import av

# --- Page config ---
st.set_page_config(
    page_title="Alarm & Noise Sound Classifier",
    page_icon="🔊",
    layout="centered",
)

# --- Styling ---
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stFileUploader>div>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("🔊 Alarm & Noise Sound Classifier")
st.markdown(
    """
    <p style='font-size:18px; color:#555;'>
    Upload a WAV file or use your microphone to classify alarm and noise sounds.
    </p>
    """,
    unsafe_allow_html=True,
)

# --- Define classes ---
CLASSES = [
    "Fire alarm", "Buzzer", "Smoke detector", "Timer alarm",
    "Opening door", "Barking", "Water", "Lawn mower"
]

# --- Dummy model ---
def dummy_train_model():
    feature_len = 26  # 13 mfcc + 12 chroma + 1 rms
    X = np.random.rand(len(CLASSES)*5, feature_len)
    y = np.repeat(CLASSES, 5)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

model = dummy_train_model()

# --- Feature extraction ---
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    rms_mean = np.mean(rms)
    return np.hstack([mfccs_mean, chroma_mean, rms_mean])

# --- Upload and classify WAV file ---
st.header("📂 Upload a WAV file to classify")
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file:
    with st.spinner("Extracting features and classifying..."):
        y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
        features = extract_features(y, sr).reshape(1, -1)

        if features.shape[1] == model.n_features_in_:
            prediction = model.predict(features)[0]
            st.success(f"✅ Predicted sound: **{prediction}**")
        else:
            st.error("⚠️ Feature size mismatch.")

st.markdown("---")

# --- Live microphone classification ---
st.header("🎤 Classify live audio from your microphone")

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
    st.success(f"🔊 Live audio prediction: **{st.session_state['live_prediction']}**")
else:
    st.info("Waiting for live audio input...")

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:12px; color:#888;'>
    Built with ❤️ using Streamlit and Librosa
    </p>
    """,
    unsafe_allow_html=True,
)
