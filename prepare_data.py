import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load extracted features
with open("features.pkl", "rb") as f:
    X = pickle.load(f)

with open("labels.pkl", "rb") as f:
    y = pickle.load(f)

# Reshape for Neural Network (CNN)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# One-hot encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
with open("train_test_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

print("✅ Data preprocessed and saved!")
