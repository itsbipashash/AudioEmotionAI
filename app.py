import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_model.h5")

# Emotion mapping
emotion_dict = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

# Function to extract MFCC features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    max_pad_length = 100
    pad_width = max_pad_length - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_length]
    return mfccs

# Streamlit App
st.title("EmoVoice: Emotion Recognition from Audio")
st.write("Upload an audio file (.wav) to predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features and predict
    features = extract_features("temp_audio.wav")
    features = features.reshape(1, 40, 100, 1)  # Reshape for CNN
    prediction = model.predict(features)
    emotion_label = np.argmax(prediction)

    # Display the result
    st.write(f"Predicted Emotion: **{emotion_dict.get(emotion_label, 'Unknown')}**")