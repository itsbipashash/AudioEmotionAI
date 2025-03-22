import os  # Import required module

# Define the dataset path
DATASET_PATH = r"C:\Users\dell\Documents\ravdess_data\audio_speech_actors_01-24"

# Check if the dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset path '{DATASET_PATH}' does not exist.")
    exit()

# Process each folder in the dataset directory
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    # Check if it's a directory
    if not os.path.isdir(folder_path):
        print(f"Skipping non-folder: {folder}")
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        print(f"Checking file: {file_path}")

        # Ensure the file is a .wav audio file
        if not file.endswith(".wav"):
            print(f"Skipping non-audio file: {file}")
            continue

        print(f"✅ Processing: {file_path}")
