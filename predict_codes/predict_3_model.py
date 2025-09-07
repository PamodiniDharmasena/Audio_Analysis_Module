import numpy as np
import librosa
import tensorflow as tf
import os

SAMPLE_RATE = 22050
TARGET_DURATION = 3  
TESTING_FOLDER = "testing_data"
FILE_EXTENSION = ".mp3"

N_MFCC = 13
N_MEL = 128
N_CHROMA = 12

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
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_length = SAMPLE_RATE * TARGET_DURATION

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y))) 
        else:
            y = y[:target_length] 

        return y, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def pad_or_truncate(feature, target_shape):
    """Ensure the feature matrix has a fixed shape."""
    if feature.shape[1] < target_shape[1]:
        pad_width = ((0, 0), (0, target_shape[1] - feature.shape[1]))
        feature = np.pad(feature, pad_width, mode='constant')
    return feature[:, :target_shape[1]]

def extract_features(file_path, model_type):
    """Extract features based on model type."""
    y, sr = load_and_pad_audio(file_path)
    if y is None:
        return None

    features = []
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))
    
    if model_type == "only_mfcc":
        return mfcc
    
    if model_type in ["only_cnn", "cnn_lstm"]:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
        mel = pad_or_truncate(mel, (N_MEL, 130))
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
        chroma = pad_or_truncate(chroma, (N_CHROMA, 130))
        
        return np.vstack([mfcc, mel, chroma])
    
    return None

def predict_audio(file_name, model_type):
    """Predict using the specified model type."""
    model_paths = {
        "only_mfcc": "./models_created/only_mfcc_audio_classification_model.h5",
        "only_cnn": "./models_created/cnn_lstm_audio_classification_model.h5",
        "cnn_lstm": "./models_created/cnn_lstm_eventdep_audio_classification_model.h5"
    }
    
    file_path = os.path.join(TESTING_FOLDER, file_name + FILE_EXTENSION)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None
    
    try:
        model = tf.keras.models.load_model(model_paths[model_type])
        
        features = extract_features(file_path, model_type)
        if features is None:
            print("Error extracting features")
            return None
        
        if model_type == "only_mfcc":
            features = np.expand_dims(features, axis=(0, -1))  
        else:
            features = np.expand_dims(features, axis=(0, -1))  
        
        predictions = model.predict(features)
        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        print(f"\n🎧 Using {model_type} model on {file_name}.mp3")
        print(f"🔹 Prediction: {LABELS[predicted_label]}")
        print(f"🔸 Confidence: {confidence:.2f}%")
        
        return LABELS[predicted_label]
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def list_test_files():
    """List all available test files in the testing folder."""
    print("\nAvailable test files:")
    files = [f.replace(FILE_EXTENSION, "") for f in os.listdir(TESTING_FOLDER) 
             if f.endswith(FILE_EXTENSION)]
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    return files

def main():
    print("🔊 Audio Classification System")
    print("=============================")
    
    models = {
        1: "only_mfcc",
        2: "only_cnn", 
        3: "cnn_lstm"
    }
    
    test_files = list_test_files()
    if not test_files:
        print("No test files found in testing_data folder!")
        return
    
    file_choice = int(input("\nEnter test file number: ")) - 1
    if file_choice < 0 or file_choice >= len(test_files):
        print("Invalid file selection")
        return
    
    print("\nAvailable models:")
    for k, v in models.items():
        print(f"{k}. {v}")
    
    model_choice = int(input("\nEnter model number: "))
    if model_choice not in models:
        print("Invalid model selection")
        return
    
    # Run prediction
    selected_file = test_files[file_choice]
    selected_model = models[model_choice]
    predict_audio(selected_file, selected_model)

if __name__ == "__main__":
    main()