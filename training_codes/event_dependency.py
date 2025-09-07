import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, LSTM, Dense, 
                                   Dropout, TimeDistributed, BatchNormalization,
                                   Bidirectional)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


DATASET_PATH = "../"
SAMPLE_RATE = 22050
TARGET_DURATION = 3
N_MFCC = 40  
N_MEL = 128
N_CHROMA = 12
SEQUENCE_LENGTH = 5
AUGMENTATION_FACTOR = 2  

LABELS = {
    "1": "Baseball Bat",
    "2": "Bomb Explosion",
    "3": "Hit and Run",
    "4": "Kill Animals",
    "5": "Lip Kissing",
    "6": "None"
}

# def load_and_pad_audio(file_path, augment=False):
#     """Enhanced audio loading with optional augmentation"""
#     try:
#         y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
#         target_length = SAMPLE_RATE * TARGET_DURATION

#         y = librosa.effects.preemphasis(y)
        
#         if len(y) < target_length:
#             padding = np.random.normal(0, 0.001, target_length - len(y))
#             y = np.concatenate([y, padding])
#         else:
#             y = y[:target_length]
        
#         if augment:
#             rate = np.random.uniform(0.9, 1.1)
#             y = librosa.effects.time_stretch(y, rate=rate)
            
#             steps = np.random.randint(-2, 2)
#             y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            
#             noise = np.random.normal(0, 0.005, len(y))
#             y = y + noise
        
#         y = librosa.util.normalize(y)
#         return y, sr
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None, None




def butter_bandpass(lowcut, highcut, fs, order=5):
    """Designs a bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=300.0, highcut=3400.0, fs=22050, order=5):
    """Applies bandpass filter to the audio signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def remove_silence(y, sr, top_db=30):
    """Removes silent segments from audio using a dB threshold."""
    intervals = librosa.effects.split(y, top_db=top_db)
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])
    return non_silent_audio

def load_and_pad_audio(file_path, augment=False):
    """Enhanced audio loading with filtering, silence removal, and optional augmentation."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_length = SAMPLE_RATE * TARGET_DURATION

        y = apply_bandpass_filter(y, lowcut=300.0, highcut=3400.0, fs=sr)

        y = remove_silence(y, sr, top_db=30)

        y = librosa.effects.preemphasis(y)

        if len(y) < target_length:
            padding = np.random.normal(0, 0.001, target_length - len(y))
            y = np.concatenate([y, padding])
        else:
            y = y[:target_length]

        if augment:
            rate = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)

            steps = np.random.randint(-2, 3)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

            noise = np.random.normal(0, 0.005, len(y))
            y = y + noise

        y = librosa.util.normalize(y)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def extract_enhanced_features(segment, sr):
    """Extract comprehensive feature set for each segment"""
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL)
    mel = librosa.power_to_db(mel, ref=np.max)
    
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=N_CHROMA)
    contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
    
    return np.vstack([mfcc, delta_mfcc, delta2_mfcc, mel, chroma, contrast])

def extract_features_with_temporal_context(file_path, augment=False):
    """Enhanced feature extraction with temporal context"""
    try:
        y, sr = load_and_pad_audio(file_path, augment=augment)
        if y is None:
            return None
            
        segment_length = len(y) // SEQUENCE_LENGTH
        features = []
        
        for i in range(SEQUENCE_LENGTH):
            segment = y[i*segment_length : (i+1)*segment_length]
            features_segment = extract_enhanced_features(segment, sr)
            features_segment = librosa.util.fix_length(
                features_segment, 
                size=130//SEQUENCE_LENGTH, 
                axis=1
            )
            features.append(features_segment)
        
        return np.array(features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset_with_augmentation():
    """Load dataset with optional augmentation"""
    X, y = [], []

    for label, category in LABELS.items():
        folder_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".mp3") or file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                
                # Original sample
                features = extract_features_with_temporal_context(file_path)
                if features is not None:
                    X.append(features)
                    y.append(int(label) - 1)
                
                # Augmented samples
                for _ in range(AUGMENTATION_FACTOR):
                    features_aug = extract_features_with_temporal_context(file_path, augment=True)
                    if features_aug is not None:
                        X.append(features_aug)
                        y.append(int(label) - 1)

    return np.array(X), np.array(y)

def create_advanced_model(input_shape, num_classes):
    """Enhanced CNN-LSTM model"""
    model = Sequential([
        # CNN feature extraction
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=l2(0.001)), input_shape=input_shape),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.25)),
        
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=l2(0.001))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.3)),
        
        TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=l2(0.001))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.35)),
        
        TimeDistributed(Flatten()),
        
        # Temporal processing
        Bidirectional(LSTM(256, return_sequences=True, 
                         dropout=0.3, recurrent_dropout=0.3)),
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)),
        
        # Classifier
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Main execution
if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset_with_augmentation()
    X = np.expand_dims(X, axis=-1)  
    y = tf.keras.utils.to_categorical(y, num_classes=len(LABELS))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Creating model...")
    model = create_advanced_model(X_train.shape[1:], len(LABELS))
    
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = dict(enumerate(class_weights))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
    ]

    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    model.save("cnn_lstm_eventdep_audio_classification_model_.h5")
    print("Model saved as enhanced_audio_classifier.h5")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Loss")
    plt.show()


