# import os
# import numpy as np
# import librosa
# import librosa.display
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Reshape, TimeDistributed, BatchNormalization
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # Paths and Constants
# DATASET_PATH = "./"  # Change this to your folder path
# SAMPLE_RATE = 22050
# TARGET_DURATION = 3  # Target length in seconds
# N_MFCC = 13  # Number of MFCC features
# N_MEL = 128  # Number of Mel Spectrogram bins
# N_CHROMA = 12  # Number of Chroma bins
# FIXED_SHAPE = (N_MFCC + N_MEL + N_CHROMA, 130)  # Ensure all features have the same shape
# SEQUENCE_LENGTH = 5  # Number of consecutive frames to consider for temporal analysis

# # Label Mapping
# LABELS = {
#     "1": "Baseball Bat",
#     "2": "Bomb Explosion",
#     "3": "Hit and Run",
#     "4": "Kill Animals",
#     "5": "Lip Kissing",
#     "6": "None"
# }

# def load_and_pad_audio(file_path):
#     """Load an audio file and pad/truncate it to a fixed duration."""
#     y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
#     target_length = SAMPLE_RATE * TARGET_DURATION

#     if len(y) < target_length:
#         y = np.pad(y, (0, target_length - len(y)))  # Pad with zeros
#     else:
#         y = y[:target_length]  # Truncate if too long

#     return y, sr

# def extract_features_with_temporal_context(file_path):
#     """Extract features with temporal context by splitting audio into segments"""
#     try:
#         y, sr = load_and_pad_audio(file_path)
        
#         # Split audio into SEQUENCE_LENGTH segments
#         segment_length = len(y) // SEQUENCE_LENGTH
#         features = []
        
#         for i in range(SEQUENCE_LENGTH):
#             segment = y[i*segment_length : (i+1)*segment_length]
            
#             # Extract features for each segment
#             mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
#             mfcc = pad_or_truncate(mfcc, (N_MFCC, 130//SEQUENCE_LENGTH))
            
#             mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL)
#             mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130//SEQUENCE_LENGTH))
            
#             chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=N_CHROMA)
#             chroma = pad_or_truncate(chroma, (N_CHROMA, 130//SEQUENCE_LENGTH))
            
#             features.append(np.vstack([mfcc, mel_spec, chroma]))
        
#         return np.array(features)

#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None

# def pad_or_truncate(feature, shape):
#     """Ensure the feature matrix has a fixed shape."""
#     pad_width = [(0, max(0, shape[0] - feature.shape[0])), (0, max(0, shape[1] - feature.shape[1]))]
#     feature = np.pad(feature, pad_width, mode='constant')
#     return feature[:, :shape[1]]  # Truncate if needed

# def load_dataset():
#     """Load the dataset and extract features with temporal context."""
#     X, y = [], []

#     for label, category in LABELS.items():
#         folder_path = os.path.join(DATASET_PATH, label)
#         if not os.path.exists(folder_path):
#             print(f"Skipping missing folder: {folder_path}")
#             continue

#         for file in os.listdir(folder_path):
#             if file.endswith(".mp3"):
#                 file_path = os.path.join(folder_path, file)
#                 features = extract_features_with_temporal_context(file_path)
#                 if features is not None:
#                     X.append(features)
#                     y.append(int(label) - 1)  # Convert label to 0-based index

#     return np.array(X), np.array(y)

# # Load and Prepare Data
# X, y = load_dataset()
# X = np.expand_dims(X, axis=-1)  # Add channel dimension for CNN
# y = tf.keras.utils.to_categorical(y, num_classes=len(LABELS))  # One-hot encode labels

# # Split Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # CNN-LSTM Model Architecture for Temporal Analysis
# model = Sequential([
#     # TimeDistributed allows CNN to process each frame independently
#     TimeDistributed(Conv2D(32, (3, 3), activation='relu')), 
#     TimeDistributed(BatchNormalization()),
#     TimeDistributed(MaxPooling2D((2, 2))),
    
#     TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
#     TimeDistributed(BatchNormalization()),
#     TimeDistributed(MaxPooling2D((2, 2))),
    
#     TimeDistributed(Conv2D(128, (3, 3), activation='relu')),
#     TimeDistributed(BatchNormalization()),
#     TimeDistributed(MaxPooling2D((2, 2))),
    
#     TimeDistributed(Flatten()),
    
#     # LSTM layers to analyze temporal sequence
#     LSTM(128, return_sequences=True, dropout=0.3),
#     LSTM(64),
    
#     # Classifier
#     Dense(128, activation='relu'),
#     Dropout(0.4),
#     Dense(len(LABELS), activation='softmax')
# ])

# # Compile Model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# # Custom callback to print validation accuracy after each epoch
# class ValAccuracyCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         val_acc = logs.get('val_accuracy')
#         print(f"Epoch {epoch+1}: Validation Accuracy = {val_acc:.4f}")

# # Train Model with callback
# history = model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=16,
#     validation_data=(X_test, y_test),
#     callbacks=[ValAccuracyCallback(), 
#               tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
# )

# # Final validation accuracy
# final_val_acc = history.history['val_accuracy'][-1]
# print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")

# # Save Model
# model.save("temporal_audio_classification_model.h5")

# # Plot Training History
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.title("Accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.title("Loss")
# plt.show()






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

# Constants with enhanced parameters
DATASET_PATH = "./"
SAMPLE_RATE = 22050
TARGET_DURATION = 3
N_MFCC = 40  # Increased from 13
N_MEL = 128
N_CHROMA = 12
SEQUENCE_LENGTH = 5
AUGMENTATION_FACTOR = 2  # Number of augmented samples per original

# Label Mapping
LABELS = {
    "1": "Baseball Bat",
    "2": "Bomb Explosion",
    "3": "Hit and Run",
    "4": "Kill Animals",
    "5": "Lip Kissing",
    "6": "None"
}

def load_and_pad_audio(file_path, augment=False):
    """Enhanced audio loading with optional augmentation"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_length = SAMPLE_RATE * TARGET_DURATION

        # Apply noise reduction
        y = librosa.effects.preemphasis(y)
        
        if len(y) < target_length:
            padding = np.random.normal(0, 0.001, target_length - len(y))
            y = np.concatenate([y, padding])
        else:
            y = y[:target_length]
        
        # Data augmentation
        if augment:
            # Time stretching
            rate = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
            
            # Pitch shifting
            steps = np.random.randint(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            
            # Add noise
            noise = np.random.normal(0, 0.005, len(y))
            y = y + noise
        
        # Normalize audio
        y = librosa.util.normalize(y)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_enhanced_features(segment, sr):
    """Extract comprehensive feature set for each segment"""
    # MFCC with delta features
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Mel spectrogram (log-scaled)
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL)
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # Chroma features
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
    # Load dataset with augmentation
    print("Loading dataset...")
    X, y = load_dataset_with_augmentation()
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    y = tf.keras.utils.to_categorical(y, num_classes=len(LABELS))

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create model
    print("Creating model...")
    model = create_advanced_model(X_train.shape[1:], len(LABELS))
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )
    model.summary()

    # Callbacks
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

    # Evaluate
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    # Save model
    model.save("enhanced_audio_classifier.h5")
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













# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (Conv2D, MaxPooling2D, LSTM, Dense, 
#                                     Dropout, Reshape, TimeDistributed, 
#                                     BatchNormalization, GlobalAveragePooling2D)
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.regularizers import l2
# from sklearn.utils import class_weight
# import matplotlib.pyplot as plt

# # Configuration
# DATASET_PATH = "./"
# SAMPLE_RATE = 22050
# TARGET_DURATION = 3  # seconds
# N_MFCC = 40
# N_MEL = 128
# N_CHROMA = 12
# SEQUENCE_LENGTH = 10
# FRAME_LENGTH = 2048
# HOP_LENGTH = 512

# # Label Mapping
# LABELS = {
#     "1": "Baseball Bat",
#     "2": "Bomb Explosion",
#     "3": "Hit and Run",
#     "4": "Kill Animals",
#     "5": "Lip Kissing",
#     "6": "None"
# }

# def load_and_preprocess_audio(file_path):
#     """Enhanced audio loading with noise reduction"""
#     try:
#         y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
#         y = librosa.effects.preemphasis(y)
#         y = librosa.util.normalize(y)
        
#         target_length = SAMPLE_RATE * TARGET_DURATION
#         if len(y) < target_length:
#             y = np.pad(y, (0, target_length - len(y)), mode='reflect')
#         else:
#             y = y[:target_length]
#         return y, sr
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None, None

# def extract_temporal_features(y, sr):
#     """Extract features with temporal context"""
#     features = []
#     segment_length = len(y) // SEQUENCE_LENGTH
#     overlap = segment_length // 2  # 50% overlap
    
#     for i in range(SEQUENCE_LENGTH):
#         start = max(0, i*segment_length - overlap)
#         end = start + segment_length + overlap
#         segment = y[start:end]
        
#         if len(segment) < 100:
#             segment = np.pad(segment, (0, 100 - len(segment)))
            
#         mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, 
#                                   n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
#         mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))
        
#         mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL,
#                                                n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
#         mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130))
        
#         chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=N_CHROMA,
#                                           n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
#         chroma = pad_or_truncate(chroma, (N_CHROMA, 130))
        
#         mfcc_delta = librosa.feature.delta(mfcc)
#         mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
#         full_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, mel_spec, chroma])
#         features.append(full_features)
    
#     return np.array(features)

# def pad_or_truncate(feature, shape):
#     """Ensure the feature matrix has a fixed shape."""
#     if feature.shape[1] < shape[1]:
#         pad_width = [(0, 0), (0, shape[1] - feature.shape[1])]
#         feature = np.pad(feature, pad_width, mode='constant')
#     else:
#         feature = feature[:, :shape[1]]
#     return feature

# def load_dataset():
#     """Improved dataset loading with balancing"""
#     X, y = [], []
    
#     for label, category in LABELS.items():
#         folder_path = os.path.join(DATASET_PATH, label)
#         if not os.path.exists(folder_path):
#             continue
            
#         files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]
#         for file in files:
#             file_path = os.path.join(folder_path, file)
#             audio, sr = load_and_preprocess_audio(file_path)
#             if audio is not None:
#                 features = extract_temporal_features(audio, sr)
#                 if features is not None:
#                     X.append(features)
#                     y.append(int(label) - 1)  # Convert to 0-based index
    
#     # Calculate class weights for imbalanced data
#     class_weights = class_weight.compute_class_weight('balanced',
#                                                     classes=np.unique(y),
#                                                     y=y)
#     class_weights = dict(enumerate(class_weights))
    
#     print("Class distribution:", np.unique(y, return_counts=True))
#     return np.array(X), np.array(y), class_weights

# # Data Loading and Preparation
# X, y, class_weights = load_dataset()
# X = np.expand_dims(X, axis=-1)  # Add channel dimension
# y = tf.keras.utils.to_categorical(y, num_classes=len(LABELS))

# # Split with stratification
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

# # Enhanced CNN-LSTM Model
# # CNN-LSTM Model with proper parentheses and architecture
# model = Sequential([
#     TimeDistributed(Conv2D(64, (3, 3), activation='relu', 
#                            kernel_regularizer=l2(0.001)),
#                     input_shape=(SEQUENCE_LENGTH, FIXED_SHAPE[0], FIXED_SHAPE[1], 1)),
#     TimeDistributed(BatchNormalization()),
#     TimeDistributed(MaxPooling2D((2, 2))),
#     TimeDistributed(Dropout(0.3)),

#     TimeDistributed(Conv2D(128, (3, 3), activation='relu', 
#                            kernel_regularizer=l2(0.001))),
#     TimeDistributed(BatchNormalization()),
#     TimeDistributed(MaxPooling2D((2, 2))),
#     TimeDistributed(Dropout(0.3)),

#     TimeDistributed(Conv2D(256, (3, 3), activation='relu',
#                            kernel_regularizer=l2(0.001))),
#     TimeDistributed(BatchNormalization()),
#     TimeDistributed(GlobalAveragePooling2D()),

#     LSTM(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.4),
#     LSTM(128, dropout=0.3, recurrent_dropout=0.3),

#     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(len(LABELS), activation='softmax')
# ])


# # Learning rate schedule
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=10000,
#     decay_rate=0.9)

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# model.compile(optimizer=optimizer,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Callbacks
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
#     tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
#     tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
# ]

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=100,
#     batch_size=32,
#     callbacks=callbacks,
#     class_weight=class_weights  # Use the computed class weights
# )

# # Evaluation
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# # Plot results
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()