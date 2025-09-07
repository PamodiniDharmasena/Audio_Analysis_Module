import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Reshape, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

DATASET_PATH = "./" 
SAMPLE_RATE = 22050
TARGET_DURATION = 3  
N_MFCC = 13
N_MEL = 128
N_CHROMA = 12
FIXED_SHAPE = (N_MFCC + N_MEL + N_CHROMA, 130)

LABELS = {
    "1": "Baseball Bat",
    "2": "Bomb Explosion",
    "3": "Hit and Run",
    "4": "Kill Animals",
    "5": "Lip Kissing",
    "6": "None"
}

def load_and_pad_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_length = SAMPLE_RATE * TARGET_DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y, sr

def extract_features(file_path):
    try:
        y, sr = load_and_pad_audio(file_path)

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
        mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
        chroma = pad_or_truncate(chroma, (N_CHROMA, 130))

        return np.vstack([mfcc, mel_spec, chroma])

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def pad_or_truncate(feature, shape):
    pad_width = [(0, max(0, shape[0] - feature.shape[0])), (0, max(0, shape[1] - feature.shape[1]))]
    feature = np.pad(feature, pad_width, mode='constant')
    return feature[:, :shape[1]]

def load_dataset():
    X, y = [], []

    for label, category in LABELS.items():
        folder_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".mp3"):
                file_path = os.path.join(folder_path, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(int(label) - 1)  

    return np.array(X), np.array(y)

X, y = load_dataset()
if len(X) == 0:
    raise ValueError("No data loaded. Please check your dataset path and structure.")

X = np.expand_dims(X, axis=-1)
y = tf.keras.utils.to_categorical(y, num_classes=len(LABELS))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: CNN + BiLSTM
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(FIXED_SHAPE[0], FIXED_SHAPE[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Reshape((16, -1)),  
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(LABELS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

class ValAccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        print(f"Epoch {epoch+1}: Validation Accuracy = {val_acc:.4f}")

# Train model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[ValAccuracyCallback()]
)

# Final accuracy
final_val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")

# Save model
model.save("cnn_lstm_eventdep_audio_classification_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Confusion matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
labels = list(LABELS.values())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
