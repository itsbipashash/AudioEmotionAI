import os
import librosa
import numpy as np
import pickle  # To save processed data

# Define dataset path
DATASET_PATH = r"C:\Users\dell\Documents\ravdess_data\audio_speech_actors_01-24"

# Function to extract features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Padding or truncating to fixed length
        max_pad_length = 100
        pad_width = max_pad_length - mfccs.shape[1]

        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_length]  # Truncate if longer

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Extract features from dataset
data = []
labels = []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue  # Skip non-folder items

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if not file.endswith(".wav"):
            continue  # Skip non-audio files

        # Extract emotion label from filename
        try:
            emotion = int(file.split("-")[2])  # Extract emotion from filename
        except ValueError:
            continue  # Skip if filename format is incorrect

        features = extract_features(file_path)
        if features is not None:
            data.append(features)
            labels.append(emotion)

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Save features and labels
with open("features.pkl", "wb") as f:
    pickle.dump(X, f)

with open("labels.pkl", "wb") as f:
    pickle.dump(y, f)

print("✅ Feature extraction complete! Features and labels saved.")
