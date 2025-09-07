
# v_c_l.py
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
import librosa

# Constants
SAMPLE_RATE = 22050
TARGET_DURATION = 3
N_MFCC = 13
N_MEL = 128
N_CHROMA = 12
FIXED_SHAPE = (N_MFCC + N_MEL + N_CHROMA, 130)

LABELS = {
    0: "Baseball Bat",
    1: "Bomb Explosion",
    2: "Hit and Run",
    3: "Kill Animals",
    4: "Lip Kissing",
    5: "None"
}

CATEGORIES = {
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
    y = np.pad(y, (0, max(0, target_length - len(y))))[:target_length]
    return y, sr

def pad_or_truncate(feature, shape):
    pad_width = [(0, max(0, shape[0] - feature.shape[0])),
                 (0, max(0, shape[1] - feature.shape[1]))]
    feature = np.pad(feature, pad_width, mode='constant')
    return feature[:, :shape[1]]

def extract_features(file_path):
    y, sr = load_and_pad_audio(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)

    mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))
    mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130))
    chroma = pad_or_truncate(chroma, (N_CHROMA, 130))

    return np.vstack([mfcc, mel_spec, chroma])

def validate_cnn_lstm_model(model_path="./models_created/cnn_lstm_eventdep_audio_classification_model.h5"):
    model = tf.keras.models.load_model(model_path)
    results = defaultdict(lambda: {"correct": 0, "total": 0})

    for folder_num, category_name in CATEGORIES.items():
        folder_path = os.path.join("validation_data", folder_num)
        if not os.path.exists(folder_path):
            continue

        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.ogg'))]
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            try:
                features = extract_features(file_path)
                features = np.expand_dims(features, axis=(0, -1))
                predictions = model.predict(features, verbose=0)
                predicted_label = np.argmax(predictions)
                actual_label = int(folder_num) - 1
                results[category_name]["total"] += 1
                if predicted_label == actual_label:
                    results[category_name]["correct"] += 1
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    return results
