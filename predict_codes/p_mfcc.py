import numpy as np
import librosa
import tensorflow as tf

# Constants (must match training script)
SAMPLE_RATE = 22050
TARGET_DURATION = 3  # 3 seconds
N_MFCC = 13
FIXED_SHAPE = (N_MFCC, 130)  # MFCC-only shape (13 coefficients x 130 frames)

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

def pad_or_truncate(feature, shape):
    """Ensure the MFCC feature matrix has a fixed shape (13, 130)."""
    # Pad if too short
    if feature.shape[1] < shape[1]:
        pad_width = ((0, 0), (0, shape[1] - feature.shape[1]))
        feature = np.pad(feature, pad_width, mode='constant')
    # Truncate if too long
    return feature[:, :shape[1]]

def extract_features(file_path):
    """Extract only MFCC features (no Mel or Chroma)."""
    y, sr = load_and_pad_audio(file_path)
    
    # Extract MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Apply padding/truncation to (13, 130)
    mfcc = pad_or_truncate(mfcc, FIXED_SHAPE)
    
    return mfcc

def predict_audio(file_path, model_path="./models_created/only_mfcc_audio_classification_model.h5"):
    """Load a trained model and classify a new audio file."""
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Extract features
    features = extract_features(file_path)
    
    if features is None:
        print("Error extracting features.")
        return None

    # Reshape for model input: (1, 13, 130, 1) 
    # (batch_size=1, height=13, width=130, channels=1)
    features = np.expand_dims(features, axis=(0, -1))  
    
    # Get prediction
    predictions = model.predict(features)
    
    # Process results
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert to percentage
    
    print(f"🔹 Prediction: {LABELS[predicted_label]} ({confidence:.2f}% confidence)")
    return LABELS[predicted_label]

# Example usage
audio_file = "./testing_data/kiss.mp3"  # Change to your file path
predict_audio(audio_file)