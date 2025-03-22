import os

# Define dataset path before using it
DATASET_PATH = r"C:\Users\dell\Documents\ravdess_data\audio_speech_actors_01-24"

# Check if the dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset path '{DATASET_PATH}' does not exist.")
    exit()

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    # Check if it is a directory
    if not os.path.isdir(folder_path):
        print(f"Skipping non-folder: {folder}")
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # Ensure it's a .wav file
        if not file.endswith(".wav"):
            print(f"Skipping non-audio file: {file}")
            continue

        # Extract emotion label from filename
        parts = file.split("-")
        if len(parts) < 3:
            print(f"Skipping file: {file} (unexpected format)")
            continue

        try:
            emotion = int(parts[2])  # Extract emotion label
        except ValueError:
            print(f"Skipping file: {file} (invalid emotion label)")
            continue

        print(f"✅ Processing: {file_path} with emotion {emotion}")


