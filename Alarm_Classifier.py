import streamlit as st
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🔊 Real-Time Alarm Sound Classifier")

# Step 1: Upload multiple WAV files with labels to train model
st.header("Step 1: Upload training data")

uploaded_train_files = st.file_uploader(
    "Upload multiple WAV files for training", 
    type=["wav"], 
    accept_multiple_files=True, 
    key="train"
)

if uploaded_train_files:
    st.write("Enter label for each training audio file:")
    train_labels = []
    for file in uploaded_train_files:
        label = st.text_input(f"Label for {file.name}", key=f"train_{file.name}")
        train_labels.append(label.strip())

    if all(train_labels) and len(train_labels) == len(uploaded_train_files):
        
        @st.cache_data(show_spinner=False)
        def extract_features(file):
            try:
                y, sr = librosa.load(file, duration=5.0, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)

                mfccs_mean = np.mean(mfccs, axis=1)
                chroma_mean = np.mean(chroma, axis=1)
                rms_mean = np.mean(rms)

                return np.hstack([mfccs_mean, chroma_mean, rms_mean])
            except Exception as e:
                st.warning(f"Error processing {file.name}: {e}")
                return None

        features = []
        valid_labels = []
        for file, label in zip(uploaded_train_files, train_labels):
            feats = extract_features(file)
            if feats is not None:
                features.append(feats)
                valid_labels.append(label)

        if len(features) < 2:
            st.error("❌ Need at least 2 valid training audio files with labels.")
        else:
            X = np.array(features)
            y = np.array(valid_labels)
            unique_classes = set(y)
            if len(unique_classes) < 2:
                st.error("❌ Need at least 2 different classes for training.")
            else:
                # Train-test split and train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if len(X_train) < 3:
                    st.error("❌ Not enough training samples for KNN with n_neighbors=3.")
                else:
                    model = KNeighborsClassifier(n_neighbors=3)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Model trained with accuracy: {acc*100:.2f}%")

                    # Save model & data to session_state
                    st.session_state['model'] = model
                    st.session_state['features'] = features
                    st.session_state['labels'] = valid_labels

else:
    st.info("Upload training WAV files to train the model.")

# Step 2: Upload one WAV file for prediction
st.header("Step 2: Upload one WAV file to classify")

uploaded_test_file = st.file_uploader("Upload a single WAV file for prediction", type=["wav"], key="test")

if uploaded_test_file:
    if 'model' not in st.session_state:
        st.error("❌ Please train the model first by uploading training data.")
    else:
        @st.cache_data(show_spinner=False)
        def extract_features_for_pred(file):
            try:
                y, sr = librosa.load(file, duration=5.0, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)

                mfccs_mean = np.mean(mfccs, axis=1)
                chroma_mean = np.mean(chroma, axis=1)
                rms_mean = np.mean(rms)

                return np.hstack([mfccs_mean, chroma_mean, rms_mean])
            except Exception as e:
                st.error(f"Error processing file: {e}")
                return None

        feats = extract_features_for_pred(uploaded_test_file)

        if feats is not None:
            model = st.session_state['model']
            feats = feats.reshape(1, -1)  # Make it 2D
            prediction = model.predict(feats)[0]
            st.success(f"Predicted Label: **{prediction}**")
