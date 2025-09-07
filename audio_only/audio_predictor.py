import numpy as np
import librosa
import tensorflow as tf

SAMPLE_RATE = 22050
TARGET_DURATION = 3 
N_MFCC = 13
N_MEL = 128
N_CHROMA = 12
FIXED_SHAPE = (N_MFCC + N_MEL + N_CHROMA, 130)  

LABELS = {
    1: "Baseball Bat",
    2: "Bomb Explosion",
    3: "Hit and Run",
    4: "Kill Animals",
    5: "Lip Kissing",
    6: "None"
}

def load_and_pad_audio(file_path):
    """Load an audio file and pad/truncate it to a fixed duration."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_length = SAMPLE_RATE * TARGET_DURATION

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))  
    else:
        y = y[:target_length]  

    return y, sr

def extract_features(file_path):
    """Extract MFCC, Mel Spectrogram, and Chroma features."""
    y, sr = load_and_pad_audio(file_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
    mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    chroma = pad_or_truncate(chroma, (N_CHROMA, 130))

    return np.vstack([mfcc, mel_spec, chroma])

def pad_or_truncate(feature, shape):
    """Ensure the feature matrix has a fixed shape."""
    pad_width = [(0, max(0, shape[0] - feature.shape[0])), (0, max(0, shape[1] - feature.shape[1]))]
    feature = np.pad(feature, pad_width, mode='constant')
    return feature[:, :shape[1]]  

def predict_audio(file_path, model_path="audio_classification_model.h5"):
    """Load a trained model and classify a new audio file."""
    model = tf.keras.models.load_model(model_path)
    features = extract_features(file_path)

    if features is None:
        print("Error extracting features.")
        return None

    features = np.expand_dims(features, axis=(0, -1))  
    predictions = model.predict(features)
    
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert to percentage
    
    # print(f"🔹 Prediction: {LABELS[predicted_label]} ({confidence:.2f}% confidence)")
    return predicted_label,confidence
#Test the model with a new audio file
# audio_file = "./20250329172658.mp3" 
# predict_audio(audio_file)


def main():
    audio_file = ""  
    predict_audio(audio_file)

if __name__ == "__main__":
    main()
