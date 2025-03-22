import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_model.h5")

# Define the path to Actor_01 folder
actor_path = r"C:\Users\dell\Documents\ravdess_data\Actor_01"

# Emotion mapping (RAVDESS dataset)
emotion_dict = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
}

# Function to extract MFCC features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Padding or truncating to fixed length
    max_pad_length = 100
    pad_width = max_pad_length - mfccs.shape[1]

    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_length]

    return mfccs

# Loop through all MP3 files in Actor_01
for file in os.listdir(actor_path):
    if file.endswith(".wav"):  # Change to ".mp3" if needed
        file_path = os.path.join(actor_path, file)

        # Extract features
        features = extract_features(file_path)
        features = features.reshape(1, 40, 100, 1)  # Reshape for CNN

        # Predict emotion
        prediction = model.predict(features)
        emotion_label = np.argmax(prediction)

        # Print result
        print(f"File: {file} → Predicted Emotion: {emotion_dict.get(emotion_label + 1, 'Unknown')}")

