import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import librosa
import joblib
import threading

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("alarm_model.pkl")  # Must match your trained model

model = load_model()
st.title("🔊 Real-Time Alarm Sound Classifier")

# For thread-safe display of prediction
result_lock = threading.Lock()
prediction_result = {"label": None}

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.buffer = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        global prediction_result
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.buffer.extend(audio)

        if len(self.buffer) >= 22050 * 5:  # 5 seconds
            y = np.array(self.buffer[:22050 * 5])
            self.buffer = self.buffer[22050 * 5:]  # keep remaining

            try:
                # Extract features
                mfccs = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13)
                chroma = librosa.feature.chroma_stft(y=y, sr=22050)
                rms = librosa.feature.rms(y=y)

                mfccs_mean = np.mean(mfccs, axis=1)
                chroma_mean = np.mean(chroma, axis=1)
                rms_mean = np.mean(rms)

                features = np.hstack([mfccs_mean, chroma_mean, rms_mean]).reshape(1, -1)

                # Predict
                pred = model.predict(features)[0]
                with result_lock:
                    prediction_result["label"] = pred
            except Exception as e:
                st.error(f"Prediction error: {e}")
        
        return frame  # No audio output

webrtc_streamer(
    key="example",
    mode="sendonly",
    in_audio=True,
    audio_processor_factory=AudioProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# Display prediction
st.subheader("🎯 Detected Sound")
label = prediction_result["label"]
if label:
    st.success(f"Prediction: **{label}**")
else:
    st.info("Listening... Make a sound.")
