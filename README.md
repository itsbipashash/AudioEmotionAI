# 🎙️ AudioEmotionAI — Emotion Recognition from Audio

> End-to-end deep learning pipeline that classifies 8 human emotions from raw `.wav` audio files using MFCC feature engineering and a Convolutional Neural Network (CNN) — built on the RAVDESS dataset.

---

## 📌 Project Overview

This project addresses a real-world NLP/audio intelligence problem: detecting human emotional states from speech without relying on text content. The system processes raw audio files, extracts structured acoustic features, trains a CNN classifier, and predicts emotions on unseen audio — covering the full ML pipeline from raw data to a deployable model.

**Emotions classified:** Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised

**Use cases:** Mental health monitoring, call centre sentiment analysis, voice assistants, HR interview analytics

---

## 🗂️ Repository Structure

```
AudioEmotionAI/
│
├── create.py             # Dataset validation — scans folders, verifies .wav files
├── organize_data.py      # Label extraction from RAVDESS filename conventions
├── extract_features.py   # MFCC feature extraction + padding + serialization
├── prepare_data.py       # Label encoding, reshaping, train/test split
├── train_model.py        # CNN architecture definition, training, model saving
├── test_model.py         # Inference on new audio files using saved model
├── app.py                # Application entry point
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Tools Used |
|---|---|
| Audio Processing | Python, Librosa |
| Feature Engineering | MFCC (Mel-Frequency Cepstral Coefficients), NumPy |
| Deep Learning | TensorFlow, Keras (Sequential CNN) |
| Data Preprocessing | Scikit-learn (label encoding, train/test split) |
| Serialization | Pickle |
| Dataset | RAVDESS (Ryerson Audio-Visual Database) |

---

## 🔍 Pipeline Walkthrough

### 1. Dataset Validation (`create.py`)
- Scans the RAVDESS dataset directory structure (24 actor folders)
- Verifies `.wav` file presence and flags missing or corrupted entries
- Ensures clean input before feature extraction begins

### 2. Label Extraction (`organize_data.py`)
- Parses RAVDESS filename conventions to extract emotion labels
- RAVDESS encodes emotion as the 3rd segment of the filename (e.g. `03-01-**05**-01-01-01-01.wav` = Angry)
- Maps numeric codes to 8 emotion classes

### 3. Feature Engineering (`extract_features.py`)
- Loads each `.wav` file using **Librosa** with `kaiser_fast` resampling for efficiency
- Extracts **40 MFCC coefficients** per audio file — capturing timbral and tonal characteristics relevant to emotion
- Applies **fixed-length padding/truncation** to 100 time steps, ensuring uniform input shape `(40, 100)` for the CNN
- Serializes extracted features and labels to `features.pkl` and `labels.pkl` using Pickle

### 4. Data Preparation (`prepare_data.py`)
- Reshapes feature arrays to `(samples, 40, 100, 1)` — adding a channel dimension for Conv2D input
- Encodes emotion labels using **one-hot encoding** (8 classes)
- Splits data into **80% training / 20% test** sets using Scikit-learn's `train_test_split`
- Saves processed splits to `train_test_data.pkl`

### 5. Model Training (`train_model.py`)

**CNN Architecture:**
```
Input: (40, 100, 1)
→ Conv2D(32 filters, 3×3, ReLU)
→ MaxPooling2D(2×2)
→ Conv2D(64 filters, 3×3, ReLU)
→ MaxPooling2D(2×2)
→ Flatten
→ Dense(128, ReLU)
→ Dropout(0.5)
→ Dense(8, Softmax)  ← output: 8 emotion classes
```

- Compiled with **Adam optimizer** (lr=0.001) and **categorical cross-entropy** loss
- Trained for **20 epochs** with batch size 32 and validation monitoring
- Saved as `emotion_model.h5` for reuse

### 6. Inference (`test_model.py`)
- Loads the saved `emotion_model.h5`
- Accepts new `.wav` files, applies the same feature extraction pipeline
- Outputs predicted emotion label per audio file

---

## 📈 Key Technical Decisions

| Decision | Reasoning |
|---|---|
| MFCC over raw waveform | MFCCs compress audio into compact, perceptually meaningful features that capture speech patterns relevant to emotion |
| Fixed padding to 100 time steps | Ensures consistent CNN input shape without losing important temporal information for most speech samples |
| Conv2D over Conv1D | Treats the MFCC matrix as a 2D image — spatial filters capture patterns across both frequency (40 coefficients) and time dimensions |
| Dropout(0.5) | Prevents overfitting given the moderate dataset size |
| One-hot encoding | Required for `categorical_crossentropy` loss with multi-class softmax output |

---

## 🚀 How to Run

**1. Install dependencies**
```bash
pip install librosa numpy scikit-learn tensorflow
```

**2. Set up dataset**

Download the [RAVDESS dataset](https://zenodo.org/record/1188976) and update `DATASET_PATH` in each script to match your local path.

**3. Run the pipeline in order**
```bash
python create.py           # Validate dataset
python organize_data.py    # Extract labels
python extract_features.py # Extract MFCC features
python prepare_data.py     # Prepare train/test data
python train_model.py      # Train CNN
python test_model.py       # Predict on new audio
```

---

## 📂 Output Files

| File | Description |
|---|---|
| `features.pkl` | Serialized MFCC feature arrays |
| `labels.pkl` | Serialized emotion labels |
| `train_test_data.pkl` | Preprocessed train/test splits |
| `emotion_model.h5` | Trained CNN model weights |

---

## 👤 Author

**Bipasha Sadhukhan**
[LinkedIn](#) · [GitHub](https://github.com/itsbipashash) · [Portfolio](#)

> *Built as part of an AI/ML internship to demonstrate end-to-end deep learning pipeline design — from raw audio ingestion through feature engineering to a trained deployable model.*

---

## 📜 Dataset Credit

RAVDESS: Livingstone SR, Russo FA (2018) *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLoS ONE. [DOI: 10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)
