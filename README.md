# AudioEmotionAI
An AI-powered emotion recognition system that analyzes audio files to detect and classify human emotions using deep learning
# Emotion Recognition from Audio Files

This project is designed to recognize emotions from audio files using a Convolutional Neural Network (CNN). The dataset used is the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains audio files of actors expressing different emotions.

## Project Structure

The project consists of the following Python scripts:

1. **create.py**: Checks the dataset path and processes each folder to identify `.wav` files.
2. **extract_features.py**: Extracts MFCC (Mel-frequency cepstral coefficients) features from the audio files and saves them along with the corresponding emotion labels.
3. **organize_data.py**: Organizes the dataset by extracting emotion labels from the filenames.
4. **prepare_data.py**: Prepares the data for training by reshaping, encoding labels, and splitting the dataset into training and testing sets.
5. **train_model.py**: Trains a CNN model on the extracted features and saves the trained model.
6. **test_model.py**: Tests the trained model on new audio files to predict emotions.

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.x
- Libraries: `librosa`, `numpy`, `scikit-learn`, `tensorflow`, `pickle`

You can install the required libraries using pip:

pip install librosa numpy scikit-learn tensorflow
Dataset
The RAVDESS dataset should be placed in the following directory:
##C:\Users\dell\Documents\ravdess_data\audio_speech_actors_01-24
If your dataset is located elsewhere, update the DATASET_PATH variable in the scripts accordingly.

Usage
Extract Features: Run extract_features.py to extract MFCC features from the audio files and save them as features.pkl and labels.pkl.
Prepare Data: Run prepare_data.py to preprocess the data and split it into training and testing sets. The processed data will be saved as train_test_data.pkl.
Train Model: Run train_model.py to train the CNN model on the preprocessed data. The trained model will be saved as emotion_model.h5.
Test Model: Run test_model.py to test the trained model on new audio files. The script will predict the emotion for each .wav file in the specified folder.
Results
The model predicts one of the following emotions for each audio file:

1: Neutral

2: Calm

3: Happy

4: Sad

5: Angry

6: Fearful

7: Disgust

8: Surprised
