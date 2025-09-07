import numpy as np
import librosa
import tensorflow as tf

# Constants (must match training script)
SAMPLE_RATE = 22050
TARGET_DURATION = 3  # 3 seconds
N_MFCC = 13
N_MEL = 128
N_CHROMA = 12
SEQUENCE_LENGTH = 5  # Must match training

# Label Mapping
LABELS = {
    0: "Baseball Bat",
    1: "Bomb Explosion",
    2: "Hit and Run",
    3: "Kill Animals",
    4: "Lip Kissing",
    5: "None"
}

def load_and_pad_audio(file_path):
    """Load an audio file and pad/truncate it to a fixed duration."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_length = SAMPLE_RATE * TARGET_DURATION

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))  # Pad with zeros
    else:
        y = y[:target_length]  # Truncate if too long

    return y, sr

def extract_features_with_temporal_context(file_path):
    """Extract features with temporal context (matches training)"""
    y, sr = load_and_pad_audio(file_path)
    
    # Split audio into SEQUENCE_LENGTH segments
    segment_length = len(y) // SEQUENCE_LENGTH
    features = []
    
    for i in range(SEQUENCE_LENGTH):
        segment = y[i*segment_length : (i+1)*segment_length]
        
        # Extract features for each segment
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
        mfcc = pad_or_truncate(mfcc, (N_MFCC, 130//SEQUENCE_LENGTH))
        
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL)
        mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130//SEQUENCE_LENGTH))
        
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=N_CHROMA)
        chroma = pad_or_truncate(chroma, (N_CHROMA, 130//SEQUENCE_LENGTH))
        
        features.append(np.vstack([mfcc, mel_spec, chroma]))
    
    return np.array(features)

def pad_or_truncate(feature, shape):
    """Ensure the feature matrix has a fixed shape."""
    pad_width = [(0, max(0, shape[0] - feature.shape[0])), 
                (0, max(0, shape[1] - feature.shape[1]))]
    feature = np.pad(feature, pad_width, mode='constant')
    return feature[:, :shape[1]]  # Truncate if needed

def predict_audio(file_path, model_path="temporal_audio_classification_model.h5"):
    """Load a trained model and classify a new audio file."""
    model = tf.keras.models.load_model(model_path)
    features = extract_features_with_temporal_context(file_path)

    if features is None:
        print("Error extracting features.")
        return None

    # Reshape features to match model input shape
    features = np.expand_dims(features, axis=-1)  # Add channel dimension
    features = np.expand_dims(features, axis=0)   # Add batch dimension
    
    predictions = model.predict(features)
    
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert to percentage
    
    print(f"🔹 Prediction: {LABELS[predicted_label]} ({confidence:.2f}% confidence)")
    return LABELS[predicted_label]

# Test the model with a new audio file
audio_file = "./testing_data/kiss2.mp3"  # Change this to your file path
predict_audio(audio_file)