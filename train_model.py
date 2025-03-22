import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
with open("train_test_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Model Architecture (CNN)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(40, 100, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')  # Output layer (emotion classes)
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save("emotion_model.h5")
print("✅ Model training complete and saved as 'emotion_model.h5'!")
