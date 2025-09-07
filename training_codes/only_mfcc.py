import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = "./"  # Path to your dataset folders
SAMPLE_RATE = 22050
TARGET_DURATION = 3  # seconds
N_MFCC = 13          # Number of MFCC coefficients
N_FFT = 2048         # FFT window size
HOP_LENGTH = 512     # Sliding window for FFT
MAX_FRAMES = 130     # Fixed number of MFCC frames

def extract_mfcc(file_path):
    """Extract MFCC features from audio file"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=TARGET_DURATION)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, 
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # Pad or truncate to fixed size
        if mfcc.shape[1] > MAX_FRAMES:
            mfcc = mfcc[:, :MAX_FRAMES]
        else:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0), (0,pad_width)), mode='constant')
            
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset():
    """Load dataset and extract MFCC features"""
    features = []
    labels = []
    
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue
            
        for file in os.listdir(label_path):
            if file.endswith(".mp3") or file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = tf.keras.utils.to_categorical(y)
    
    return X, y, le.classes_

# Load data
X, y, class_names = load_dataset()

# Add channel dimension (for CNN)
X = np.expand_dims(X, axis=-1)  # Shape: (samples, N_MFCC, MAX_FRAMES, 1)
print(f"Input shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build MFCC-based CNN model with adjusted architecture
model = Sequential([
    # First conv block
    Conv2D(32, (3, 3), activation='relu', input_shape=(N_MFCC, MAX_FRAMES, 1)),
    BatchNormalization(),
    MaxPooling2D((1, 2)),  # Reduced pooling to preserve dimensions
    
    # Second conv block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((1, 2)),  # Only pool along time dimension
    
    # Third conv block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((1, 2)),  # Only pool along time dimension
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.save("only_mfcc_audio_classification_model.h5")

# Train model
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()