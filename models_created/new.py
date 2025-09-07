# # # # import os
# # # # import numpy as np
# # # # import librosa
# # # # import tensorflow as tf
# # # # import matplotlib.pyplot as plt
# # # # from sklearn.metrics import confusion_matrix, classification_report
# # # # import seaborn as sns

# # # # # Constants (must match training scripts)
# # # # SAMPLE_RATE = 22050
# # # # TARGET_DURATION = 3  # 3 seconds
# # # # N_MFCC = 13
# # # # N_MEL = 128
# # # # N_CHROMA = 12
# # # # FIXED_SHAPE = (N_MFCC + N_MEL + N_CHROMA, 130)

# # # # # Label Mapping
# # # # LABELS = {
# # # #     0: "Baseball Bat",
# # # #     1: "Bomb Explosion",
# # # #     2: "Hit and Run",
# # # #     3: "Kill Animals",
# # # #     4: "Lip Kissing",
# # # #     5: "None"
# # # # }

# # # # def load_and_pad_audio(file_path):
# # # #     """Load an audio file and pad/truncate it to a fixed duration."""
# # # #     y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
# # # #     target_length = SAMPLE_RATE * TARGET_DURATION

# # # #     if len(y) < target_length:
# # # #         y = np.pad(y, (0, target_length - len(y)))  # Pad with zeros
# # # #     else:
# # # #         y = y[:target_length]  # Truncate if too long

# # # #     return y, sr

# # # # def extract_features(file_path):
# # # #     """Extract MFCC, Mel Spectrogram, and Chroma features."""
# # # #     try:
# # # #         y, sr = load_and_pad_audio(file_path)

# # # #         # Extract MFCC
# # # #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# # # #         mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))

# # # #         # Extract Mel Spectrogram
# # # #         mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
# # # #         mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130))

# # # #         # Extract Chroma
# # # #         chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
# # # #         chroma = pad_or_truncate(chroma, (N_CHROMA, 130))

# # # #         # Stack features
# # # #         return np.vstack([mfcc, mel_spec, chroma])
# # # #     except Exception as e:
# # # #         print(f"Error processing {file_path}: {e}")
# # # #         return None

# # # # def pad_or_truncate(feature, shape):
# # # #     """Ensure the feature matrix has a fixed shape."""
# # # #     pad_width = [(0, max(0, shape[0] - feature.shape[0])), (0, max(0, shape[1] - feature.shape[1]))]
# # # #     feature = np.pad(feature, pad_width, mode='constant')
# # # #     return feature[:, :shape[1]]  # Truncate if needed

# # # # def load_validation_data(validation_path="./validation_data"):
# # # #     """Load the validation dataset."""
# # # #     X_val, y_val = [], []
    
# # # #     for label in LABELS.keys():
# # # #         folder_path = os.path.join(validation_path, str(int(label)+1))  # Folders are named 1-6
# # # #         if not os.path.exists(folder_path):
# # # #             print(f"Warning: Missing validation folder for label {label}")
# # # #             continue
            
# # # #         for file in os.listdir(folder_path):
# # # #             if file.endswith(".mp3"):
# # # #                 file_path = os.path.join(folder_path, file)
# # # #                 features = extract_features(file_path)
# # # #                 if features is not None:
# # # #                     X_val.append(features)
# # # #                     y_val.append(int(label))  # 0-based label
                    
# # # #     return np.array(X_val), np.array(y_val)

# # # # def evaluate_model(model, X_val, y_val, model_name):
# # # #     """Evaluate a model and return metrics."""
# # # #     # Prepare data
# # # #     X_val_expanded = np.expand_dims(X_val, axis=-1)
# # # #     y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(LABELS))
    
# # # #     # Evaluate
# # # #     loss, accuracy = model.evaluate(X_val_expanded, y_val_onehot, verbose=0)
    
# # # #     # Predictions
# # # #     y_pred_probs = model.predict(X_val_expanded)
# # # #     y_pred = np.argmax(y_pred_probs, axis=1)
    
# # # #     # Confusion matrix
# # # #     cm = confusion_matrix(y_val, y_pred)
    
# # # #     # Classification report
# # # #     report = classification_report(y_val, y_pred, target_names=LABELS.values(), output_dict=True)
    
# # # #     return {
# # # #         'accuracy': accuracy,
# # # #         'loss': loss,
# # # #         'confusion_matrix': cm,
# # # #         'classification_report': report
# # # #     }

# # # # def plot_comparison(results):
# # # #     """Plot comparison between models."""
# # # #     models = list(results.keys())
# # # #     accuracies = [results[m]['accuracy'] for m in models]
    
# # # #     plt.figure(figsize=(10, 6))
# # # #     bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    
# # # #     plt.title('Model Comparison on Validation Set')
# # # #     plt.xlabel('Model Architecture')
# # # #     plt.ylabel('Accuracy')
# # # #     plt.ylim(0, 1.0)
    
# # # #     # Add values on top of bars
# # # #     for bar in bars:
# # # #         height = bar.get_height()
# # # #         plt.text(bar.get_x() + bar.get_width()/2., height,
# # # #                  f'{height:.4f}',
# # # #                  ha='center', va='bottom')
    
# # # #     plt.tight_layout()
# # # #     plt.show()

# # # # def plot_confusion_matrices(results):
# # # #     """Plot confusion matrices for both models."""
# # # #     fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
# # # #     for idx, (model_name, result) in enumerate(results.items()):
# # # #         cm = result['confusion_matrix']
# # # #         sns.heatmap(cm, annot=True, fmt='d', 
# # # #                     xticklabels=LABELS.values(), 
# # # #                     yticklabels=LABELS.values(),
# # # #                     cmap='Blues', ax=axes[idx])
# # # #         axes[idx].set_title(f'Confusion Matrix - {model_name}')
# # # #         axes[idx].set_xlabel('Predicted')
# # # #         axes[idx].set_ylabel('True')
# # # #         axes[idx].tick_params(axis='both', which='major', labelsize=8)
    
# # # #     plt.tight_layout()
# # # #     plt.show()

# # # # def main():
# # # #     # Load validation data
# # # #     print("Loading validation data...")
# # # #     X_val, y_val = load_validation_data()
    
# # # #     if len(X_val) == 0:
# # # #         print("No validation data found. Please check your validation folder structure.")
# # # #         return
    
# # # #     print(f"Loaded {len(X_val)} validation samples.")
    
# # # #     # Load both models
# # # #     print("Loading models...")
# # # #     try:
# # # #         cnn_lstm_model = tf.keras.models.load_model("cnn_lstm_eventdep_audio_classification_model.h5")
# # # #         cnn_bilstm_model = tf.keras.models.load_model("vhghgncnn_bilstm_audio_classification_model.h5")
# # # #     except Exception as e:
# # # #         print(f"Error loading models: {e}")
# # # #         return
    
# # # #     # Evaluate models
# # # #     results = {}
# # # #     print("\nEvaluating CNN+LSTM model...")
# # # #     results['CNN+LSTM'] = evaluate_model(cnn_lstm_model, X_val, y_val, "CNN+LSTM")
    
# # # #     print("\nEvaluating CNN+BiLSTM model...")
# # # #     results['CNN+BiLSTM'] = evaluate_model(cnn_bilstm_model, X_val, y_val, "CNN+BiLSTM")
    
# # # #     # Print results
# # # #     print("\n=== Evaluation Results ===")
# # # #     for model_name, metrics in results.items():
# # # #         print(f"\n{model_name}:")
# # # #         print(f"Accuracy: {metrics['accuracy']:.4f}")
# # # #         print(f"Loss: {metrics['loss']:.4f}")
# # # #         print("\nClassification Report:")
# # # #         print(classification_report(y_val, 
# # # #                                   np.argmax(results[model_name]['confusion_matrix'], axis=1),
# # # #                                   target_names=LABELS.values()))
    
# # # #     # Visualizations
# # # #     plot_comparison(results)
# # # #     plot_confusion_matrices(results)

# # # # if __name__ == "__main__":
# # # #     main()



# # # import os
# # # import numpy as np
# # # import librosa
# # # import tensorflow as tf
# # # import matplotlib.pyplot as plt
# # # from sklearn.metrics import confusion_matrix, classification_report
# # # import seaborn as sns

# # # # Constants (must match training scripts)
# # # SAMPLE_RATE = 22050
# # # TARGET_DURATION = 3  # 3 seconds
# # # N_MFCC = 13
# # # N_MEL = 128
# # # N_CHROMA = 12
# # # FIXED_SHAPE = (N_MFCC + N_MEL + N_CHROMA, 130)

# # # # Label Mapping
# # # LABELS = {
# # #     0: "Baseball Bat",
# # #     1: "Bomb Explosion",
# # #     2: "Hit and Run",
# # #     3: "Kill Animals",
# # #     4: "Lip Kissing",
# # #     5: "None"
# # # }

# # # def load_and_pad_audio(file_path):
# # #     """Load an audio file and pad/truncate it to a fixed duration."""
# # #     y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
# # #     target_length = SAMPLE_RATE * TARGET_DURATION

# # #     if len(y) < target_length:
# # #         y = np.pad(y, (0, target_length - len(y)))  # Pad with zeros
# # #     else:
# # #         y = y[:target_length]  # Truncate if too long

# # #     return y, sr

# # # def extract_features(file_path):
# # #     """Extract MFCC, Mel Spectrogram, and Chroma features."""
# # #     try:
# # #         y, sr = load_and_pad_audio(file_path)

# # #         # Extract MFCC
# # #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# # #         mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))

# # #         # Extract Mel Spectrogram
# # #         mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
# # #         mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130))

# # #         # Extract Chroma
# # #         chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
# # #         chroma = pad_or_truncate(chroma, (N_CHROMA, 130))

# # #         # Stack features
# # #         return np.vstack([mfcc, mel_spec, chroma])
# # #     except Exception as e:
# # #         print(f"Error processing {file_path}: {e}")
# # #         return None

# # # def pad_or_truncate(feature, shape):
# # #     """Ensure the feature matrix has a fixed shape."""
# # #     pad_width = [(0, max(0, shape[0] - feature.shape[0])), (0, max(0, shape[1] - feature.shape[1]))]
# # #     feature = np.pad(feature, pad_width, mode='constant')
# # #     return feature[:, :shape[1]]  # Truncate if needed

# # # def load_validation_data(validation_path="../validation_data"):
# # #     """Load the validation dataset."""
# # #     X_val, y_val = [], []
    
# # #     for label in LABELS.keys():
# # #         folder_path = os.path.join(validation_path, str(int(label)+1))  # Folders are named 1-6
# # #         if not os.path.exists(folder_path):
# # #             print(f"Warning: Missing validation folder for label {label}")
# # #             continue
            
# # #         for file in os.listdir(folder_path):
# # #             if file.endswith((".mp3", ".wav")):  # Accept both mp3 and wav files
# # #                 file_path = os.path.join(folder_path, file)
# # #                 features = extract_features(file_path)
# # #                 if features is not None:
# # #                     X_val.append(features)
# # #                     y_val.append(int(label))  # 0-based label
                    
# # #     return np.array(X_val), np.array(y_val)

# # # def find_model_file(model_name):
# # #     """Search for model file in common locations."""
# # #     possible_locations = [
# # #         os.path.join(os.getcwd(), model_name),
# # #         os.path.join(os.getcwd(), "models", model_name),
# # #         os.path.join(os.getcwd(), "saved_models", model_name),
# # #         os.path.join(os.path.dirname(os.getcwd()), model_name),
# # #     ]
    
# # #     for path in possible_locations:
# # #         if os.path.exists(path):
# # #             return path
    
# # #     print(f"\nCould not find {model_name} in these locations:")
# # #     for loc in possible_locations:
# # #         print(f"- {loc}")
# # #     return None

# # # def load_model(model_name):
# # #     """Attempt to load a model with helpful error messages."""
# # #     model_path = find_model_file(model_name)
# # #     if model_path is None:
# # #         print(f"\nPlease ensure the model file '{model_name}' exists in one of these locations:")
# # #         print(f"1. Current directory: {os.getcwd()}")
# # #         print(f"2. 'models' subdirectory")
# # #         print(f"3. 'saved_models' subdirectory")
# # #         print(f"4. Parent directory")
# # #         return None
    
# # #     try:
# # #         model = tf.keras.models.load_model(model_path)
# # #         print(f"Successfully loaded model from: {model_path}")
# # #         return model
# # #     except Exception as e:
# # #         print(f"\nError loading model from {model_path}:")
# # #         print(str(e))
# # #         print("\nPossible solutions:")
# # #         print("1. Check if the model file is complete and not corrupted")
# # #         print("2. Verify your TensorFlow version matches the training environment")
# # #         print("3. Ensure all required dependencies are installed")
# # #         return None

# # # def evaluate_model(model, X_val, y_val, model_name):
# # #     """Evaluate a model and return metrics."""
# # #     if model is None:
# # #         return None
        
# # #     # Prepare data
# # #     X_val_expanded = np.expand_dims(X_val, axis=-1)
# # #     y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(LABELS))
    
# # #     # Debugging: Print shapes before evaluation
# # #     print(f"\nDebugging {model_name}:")
# # #     print(f"X_val shape: {X_val.shape}")
# # #     print(f"X_val_expanded shape: {X_val_expanded.shape}")
# # #     print(f"y_val shape: {y_val.shape}")
# # #     print(f"y_val_onehot shape: {y_val_onehot.shape}")
    
# # #     try:
# # #         # Evaluate
# # #         loss, accuracy = model.evaluate(X_val_expanded, y_val_onehot, verbose=0)
        
# # #         # Predictions
# # #         y_pred_probs = model.predict(X_val_expanded, verbose=0)
# # #         y_pred = np.argmax(y_pred_probs, axis=1)
        
# # #         # Debugging: Print prediction shapes
# # #         print(f"y_pred_probs shape: {y_pred_probs.shape}")
# # #         print(f"y_pred shape: {y_pred.shape}")
        
# # #         # Confusion matrix
# # #         cm = confusion_matrix(y_val, y_pred)
        
# # #         # Classification report
# # #         report = classification_report(y_val, y_pred, target_names=LABELS.values(), output_dict=True)
        
# # #         return {
# # #             'accuracy': accuracy,
# # #             'loss': loss,
# # #             'confusion_matrix': cm,
# # #             'classification_report': report
# # #         }
# # #     except Exception as e:
# # #         print(f"\nError during evaluation of {model_name}:")
# # #         print(str(e))
# # #         print("\nPossible causes:")
# # #         print("1. Model output shape doesn't match expected number of classes")
# # #         print("2. Data preprocessing inconsistency")
# # #         print("3. Model architecture mismatch")
# # #         return None

# # # def plot_comparison(results):
# # #     """Plot comparison between models."""
# # #     # Filter out None results (failed models)
# # #     valid_results = {k: v for k, v in results.items() if v is not None}
# # #     if not valid_results:
# # #         print("\nNo valid model results to plot")
# # #         return
        
# # #     models = list(valid_results.keys())
# # #     accuracies = [valid_results[m]['accuracy'] for m in models]
    
# # #     plt.figure(figsize=(10, 6))
# # #     bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    
# # #     plt.title('Model Comparison on Validation Set')
# # #     plt.xlabel('Model Architecture')
# # #     plt.ylabel('Accuracy')
# # #     plt.ylim(0, 1.0)
    
# # #     # Add values on top of bars
# # #     for bar in bars:
# # #         height = bar.get_height()
# # #         plt.text(bar.get_x() + bar.get_width()/2., height,
# # #                  f'{height:.4f}',
# # #                  ha='center', va='bottom')
    
# # #     plt.tight_layout()
# # #     plt.show()

# # # def plot_confusion_matrices(results):
# # #     """Plot confusion matrices for both models."""
# # #     # Filter out None results (failed models)
# # #     valid_results = {k: v for k, v in results.items() if v is not None}
# # #     if not valid_results:
# # #         print("\nNo valid model results to plot")
# # #         return
        
# # #     fig, axes = plt.subplots(1, len(valid_results), figsize=(18, 7))
# # #     if len(valid_results) == 1:  # Handle case where only one model loaded
# # #         axes = [axes]
    
# # #     for idx, (model_name, result) in enumerate(valid_results.items()):
# # #         cm = result['confusion_matrix']
# # #         sns.heatmap(cm, annot=True, fmt='d', 
# # #                     xticklabels=LABELS.values(), 
# # #                     yticklabels=LABELS.values(),
# # #                     cmap='Blues', ax=axes[idx])
# # #         axes[idx].set_title(f'Confusion Matrix - {model_name}')
# # #         axes[idx].set_xlabel('Predicted')
# # #         axes[idx].set_ylabel('True')
# # #         axes[idx].tick_params(axis='both', which='major', labelsize=8)
    
# # #     plt.tight_layout()
# # #     plt.show()

# # # def main():
# # #     print("\n=== Audio Classification Model Evaluation ===")
    
# # #     # Load validation data
# # #     print("\n[1/3] Loading validation data...")
# # #     X_val, y_val = load_validation_data()
    
# # #     if len(X_val) == 0:
# # #         print("\nError: No validation data found.")
# # #         print("Please check your validation folder structure:")
# # #         print("- Expected structure: validation_data/1/, validation_data/2/, etc.")
# # #         print(f"- Current working directory: {os.getcwd()}")
# # #         return
    
# # #     print(f"\nSuccessfully loaded {len(X_val)} validation samples.")
    
# # #     # Define model names
# # #     model_files = {
# # #         'CNN+LSTM': "cnn_lstm_eventdep_audio_classification_model.h5",
# # #         'CNN+BiLSTM': "vhghgncnn_bilstm_audio_classification_model.h5"
# # #     }
    
# # #     # Load models
# # #     print("\n[2/3] Loading models...")
# # #     models = {}
# # #     for name, filename in model_files.items():
# # #         print(f"\nAttempting to load {name} model...")
# # #         models[name] = load_model(filename)
    
# # #     # Evaluate models
# # #     print("\n[3/3] Evaluating models...")
# # #     results = {}
# # #     for name, model in models.items():
# # #         if model is not None:
# # #             print(f"\nEvaluating {name} model...")
# # #             results[name] = evaluate_model(model, X_val, y_val, name)
# # #         else:
# # #             print(f"\nSkipping evaluation for {name} - model not loaded")
# # #             results[name] = None
    
# # #     # Print results
# # #     print("\n=== Evaluation Results ===")
# # #     for model_name, metrics in results.items():
# # #         if metrics is None:
# # #             continue
# # #         print(f"\n{model_name} Results:")
# # #         print(f"- Accuracy: {metrics['accuracy']:.4f}")
# # #         print(f"- Loss: {metrics['loss']:.4f}")
# # #         print("\nClassification Report:")
# # #         print(classification_report(y_val, 
# # #                                   np.argmax(metrics['confusion_matrix'], axis=1),
# # #                                   target_names=LABELS.values()))
    
# # #     # Visualizations
# # #     plot_comparison(results)
# # #     plot_confusion_matrices(results)

# # #     print("\nEvaluation complete!")

# # # if __name__ == "__main__":
# # #     main()


# # import os
# # import numpy as np
# # import librosa
# # import tensorflow as tf
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import confusion_matrix, classification_report
# # import seaborn as sns

# # # Constants (must match training scripts)
# # SAMPLE_RATE = 22050
# # TARGET_DURATION = 3  # 3 seconds
# # N_MFCC = 13
# # N_MEL = 128
# # N_CHROMA = 12
# # FIXED_SHAPE = (N_MFCC + N_MEL + N_CHROMA, 130)

# # # Label Mapping
# # LABELS = {
# #     0: "Baseball Bat",
# #     1: "Bomb Explosion",
# #     2: "Hit and Run",
# #     3: "Kill Animals",
# #     4: "Lip Kissing",
# #     5: "None"
# # }

# # def load_and_pad_audio(file_path):
# #     """Load an audio file and pad/truncate it to a fixed duration."""
# #     y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
# #     target_length = SAMPLE_RATE * TARGET_DURATION

# #     if len(y) < target_length:
# #         y = np.pad(y, (0, target_length - len(y)))  # Pad with zeros
# #     else:
# #         y = y[:target_length]  # Truncate if too long

# #     return y, sr

# # def extract_features(file_path):
# #     """Extract MFCC, Mel Spectrogram, and Chroma features."""
# #     try:
# #         y, sr = load_and_pad_audio(file_path)

# #         # Extract MFCC
# #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# #         mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))

# #         # Extract Mel Spectrogram
# #         mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
# #         mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130))

# #         # Extract Chroma
# #         chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
# #         chroma = pad_or_truncate(chroma, (N_CHROMA, 130))

# #         # Stack features
# #         return np.vstack([mfcc, mel_spec, chroma])
# #     except Exception as e:
# #         print(f"Error processing {file_path}: {e}")
# #         return None

# # def pad_or_truncate(feature, shape):
# #     """Ensure the feature matrix has a fixed shape."""
# #     pad_width = [(0, max(0, shape[0] - feature.shape[0])), (0, max(0, shape[1] - feature.shape[1]))]
# #     feature = np.pad(feature, pad_width, mode='constant')
# #     return feature[:, :shape[1]]  # Truncate if needed

# # def load_validation_data(validation_path="../validation_data"):
# #     """Load the validation dataset."""
# #     X_val, y_val = [], []
    
# #     for label in LABELS.keys():
# #         folder_path = os.path.join(validation_path, str(int(label)+1))  # Folders are named 1-6
# #         if not os.path.exists(folder_path):
# #             print(f"Warning: Missing validation folder for label {label}")
# #             continue
            
# #         for file in os.listdir(folder_path):
# #             if file.endswith((".mp3", ".wav")):  # Accept both mp3 and wav files
# #                 file_path = os.path.join(folder_path, file)
# #                 features = extract_features(file_path)
# #                 if features is not None:
# #                     X_val.append(features)
# #                     y_val.append(int(label))  # 0-based label
                    
# #     return np.array(X_val), np.array(y_val)

# # def load_model(model_path):
# #     """Load a trained model with error handling."""
# #     try:
# #         model = tf.keras.models.load_model(model_path)
# #         print(f"Successfully loaded model from: {model_path}")
# #         return model
# #     except Exception as e:
# #         print(f"Error loading model from {model_path}: {e}")
# #         return None

# # def evaluate_model(model, X_val, y_val, model_name):
# #     """Evaluate a model and return metrics."""
# #     if model is None:
# #         return None
        
# #     # Prepare data
# #     X_val_expanded = np.expand_dims(X_val, axis=-1)
# #     y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(LABELS))
    
# #     # Debug shapes
# #     print(f"\nShapes for {model_name}:")
# #     print(f"X_val_expanded: {X_val_expanded.shape}")
# #     print(f"y_val: {y_val.shape}")
# #     print(f"y_val_onehot: {y_val_onehot.shape}")
    
# #     # Evaluate
# #     loss, accuracy = model.evaluate(X_val_expanded, y_val_onehot, verbose=0)
    
# #     # Predictions
# #     y_pred_probs = model.predict(X_val_expanded, verbose=0)
# #     y_pred = np.argmax(y_pred_probs, axis=1)
    
# #     print(f"y_pred shape: {y_pred.shape}")
    
# #     # Confusion matrix
# #     cm = confusion_matrix(y_val, y_pred)
    
# #     # Classification report
# #     report = classification_report(y_val, y_pred, target_names=LABELS.values(), output_dict=True)
    
# #     return {
# #         'accuracy': accuracy,
# #         'loss': loss,
# #         'confusion_matrix': cm,
# #         'predictions': y_pred,
# #         'true_labels': y_val
# #     }

# # def plot_prediction_distribution(y_true, y_pred, model_name):
# #     """Plot the distribution of predictions across categories."""
# #     # Count predictions per category
# #     unique, counts = np.unique(y_pred, return_counts=True)
# #     pred_counts = dict(zip(unique, counts))
    
# #     # Fill in missing categories with 0
# #     for i in range(len(LABELS)):
# #         if i not in pred_counts:
# #             pred_counts[i] = 0
    
# #     # Sort by label index
# #     sorted_labels = [LABELS[i] for i in sorted(pred_counts.keys())]
# #     sorted_counts = [pred_counts[i] for i in sorted(pred_counts.keys())]
    
# #     # Create bar plot
# #     plt.figure(figsize=(10, 6))
# #     bars = plt.bar(sorted_labels, sorted_counts, color='skyblue')
    
# #     plt.title(f'Prediction Distribution - {model_name}')
# #     plt.xlabel('Categories')
# #     plt.ylabel('Number of Predictions')
# #     plt.xticks(rotation=45)
    
# #     # Add counts on top of bars
# #     for bar in bars:
# #         height = bar.get_height()
# #         plt.text(bar.get_x() + bar.get_width()/2., height,
# #                  f'{int(height)}',
# #                  ha='center', va='bottom')
    
# #     plt.tight_layout()
# #     plt.show()

# # def main():
# #     # Load validation data
# #     print("Loading validation data...")
# #     X_val, y_val = load_validation_data()
    
# #     if len(X_val) == 0:
# #         print("No validation data found. Please check your validation folder structure.")
# #         return
    
# #     print(f"Loaded {len(X_val)} validation samples.")
    
# #     # Load models
# #     model_paths = {
# #         'CNN+LSTM': "cnn_lstm_eventdep_audio_classification_model.h5",
# #         'CNN+BiLSTM': "vhghgncnn_bilstm_audio_classification_model.h5"
# #     }
    
# #     results = {}
# #     for name, path in model_paths.items():
# #         print(f"\nLoading {name} model...")
# #         model = load_model(path)
# #         if model is not None:
# #             results[name] = evaluate_model(model, X_val, y_val, name)
# #             if results[name] is not None:
# #                 plot_prediction_distribution(results[name]['true_labels'], 
# #                                           results[name]['predictions'], 
# #                                           name)
    
# #     # Print results
# #     print("\n=== Evaluation Results ===")
# #     for model_name, metrics in results.items():
# #         if metrics is None:
# #             continue
# #         print(f"\n{model_name}:")
# #         print(f"Accuracy: {metrics['accuracy']:.4f}")
# #         print(f"Loss: {metrics['loss']:.4f}")
# #         print("\nClassification Report:")
# #         print(classification_report(metrics['true_labels'], 
# #                                   metrics['predictions'],
# #                                   target_names=LABELS.values()))

# # if __name__ == "__main__":
# #     main()


# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns

# # Constants
# SAMPLE_RATE = 22050
# TARGET_DURATION = 3
# N_MFCC = 13
# N_MEL = 128
# N_CHROMA = 12
# FIXED_SHAPE = (N_MFCC + N_MEL + N_CHROMA, 130)

# # Label Mapping
# LABELS = {
#     0: "Baseball Bat",
#     1: "Bomb Explosion",
#     2: "Hit and Run",
#     3: "Kill Animals",
#     4: "Lip Kissing",
#     5: "None"
# }

# def load_and_pad_audio(file_path):
#     y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
#     target_length = SAMPLE_RATE * TARGET_DURATION
#     if len(y) < target_length:
#         y = np.pad(y, (0, target_length - len(y)))
#     else:
#         y = y[:target_length]
#     return y, sr

# def extract_features(file_path):
#     try:
#         y, sr = load_and_pad_audio(file_path)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
#         mfcc = pad_or_truncate(mfcc, (N_MFCC, 130))
#         mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
#         mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130))
#         chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
#         chroma = pad_or_truncate(chroma, (N_CHROMA, 130))
#         return np.vstack([mfcc, mel_spec, chroma])
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None

# def pad_or_truncate(feature, shape):
#     pad_width = [(0, max(0, shape[0] - feature.shape[0])), (0, max(0, shape[1] - feature.shape[1]))]
#     feature = np.pad(feature, pad_width, mode='constant')
#     return feature[:, :shape[1]]

# def load_validation_data(validation_path="../validation_data"):
#     X_val, y_val = [], []
#     for label in LABELS.keys():
#         folder_path = os.path.join(validation_path, str(int(label)+1))
#         if not os.path.exists(folder_path):
#             continue
#         for file in os.listdir(folder_path):
#             if file.endswith((".mp3", ".wav")):
#                 file_path = os.path.join(folder_path, file)
#                 features = extract_features(file_path)
#                 if features is not None:
#                     X_val.append(features)
#                     y_val.append(int(label))
#     return np.array(X_val), np.array(y_val)

# def load_model(model_path):
#     try:
#         model = tf.keras.models.load_model(model_path)
#         print(f"Loaded model: {model_path}")
#         return model
#     except Exception as e:
#         print(f"Error loading {model_path}: {e}")
#         return None

# def evaluate_model(model, X_val, y_val):
#     if model is None:
#         return None
        
#     X_val_expanded = np.expand_dims(X_val, axis=-1)
#     y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(LABELS))
    
#     loss, accuracy = model.evaluate(X_val_expanded, y_val_onehot, verbose=0)
#     y_pred = np.argmax(model.predict(X_val_expanded, verbose=0), axis=1)
    
#     return {
#         'accuracy': accuracy,
#         'predictions': y_pred,
#         'true_labels': y_val
#     }

# def plot_combined_results(results):
#     if not results:
#         print("No results to plot")
#         return
    
#     # Prepare data
#     models = list(results.keys())
#     accuracies = [results[m]['accuracy']*100 for m in models]  # Convert to percentage
    
#     # Create figure
#     plt.figure(figsize=(12, 6))
    
#     # Bar plot for accuracy comparison
#     bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
    
#     # Add percentage labels
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.1f}%',
#                 ha='center', va='bottom', fontsize=12)
    
#     # Formatting
#     plt.title('Model Accuracy Comparison', fontsize=14)
#     plt.xlabel('Model Architecture', fontsize=12)
#     plt.ylabel('Accuracy (%)', fontsize=12)
#     plt.ylim(0, 100)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.show()

# def main():
#     # Load data
#     X_val, y_val = load_validation_data()
#     if len(X_val) == 0:
#         print("No validation data found")
#         return
    
#     # Load and evaluate models
#     model_paths = {
#         'CNN+LSTM': "cnn_lstm_eventdep_audio_classification_model.h5",
#         'CNN+BiLSTM': "vhghgncnn_bilstm_audio_classification_model.h5"
#     }
    
#     results = {}
#     for name, path in model_paths.items():
#         model = load_model(path)
#         if model:
#             results[name] = evaluate_model(model, X_val, y_val)
    
#     # Plot combined results
#     plot_combined_results(results)
    
#     # Print detailed reports
#     for model_name, metrics in results.items():
#         if metrics:
#             print(f"\n{model_name} Classification Report:")
#             print(classification_report(metrics['true_labels'], 
#                                      metrics['predictions'],
#                                      target_names=LABELS.values()))

# if __name__ == "__main__":
#     main()


import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

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

def load_validation_data(validation_path="../validation_data"):
    X_val, y_val = [], []
    for label in LABELS.keys():
        folder_path = os.path.join(validation_path, str(int(label)+1))
        if not os.path.exists(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith((".mp3", ".wav")):
                file_path = os.path.join(folder_path, file)
                features = extract_features(file_path)
                if features is not None:
                    X_val.append(features)
                    y_val.append(int(label))
    return np.array(X_val), np.array(y_val)

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def calculate_category_accuracies(y_true, y_pred):
    category_acc = []
    for label in sorted(LABELS.keys()):
        idx = (y_true == label)
        if sum(idx) > 0:
            acc = accuracy_score(y_true[idx], y_pred[idx]) * 100
        else:
            acc = 0
        category_acc.append(acc)
    return category_acc

def plot_comprehensive_results(results):
    if not results:
        print("No results to plot")
        return
    
    models = list(results.keys())
    categories = list(LABELS.values())
    
    # Prepare data
    category_acc = {model: calculate_category_accuracies(results[model]['true_labels'], 
                    results[model]['predictions']) 
                   for model in models}
    overall_acc = {model: results[model]['accuracy']*100 for model in models}
    
    plt.figure(figsize=(14, 8))
    
    bar_width = 0.35
    x = np.arange(len(categories))
    
    for i, model in enumerate(models):
        offset = bar_width * i
        bars = plt.bar(x + offset, category_acc[model], width=bar_width, 
                      label=f'{model} (Per Category)')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    for i, model in enumerate(models):
        plt.axhline(y=overall_acc[model], color=['#1f77b4', '#ff7f0e'][i], 
                   linestyle='--', linewidth=2,
                   label=f'{model} (Overall: {overall_acc[model]:.1f}%)')
    
    # Formatting
    plt.title('Model Performance by Category and Overall', fontsize=14)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(x + bar_width/2, categories, rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def main():
    X_val, y_val = load_validation_data()
    if len(X_val) == 0:
        print("No validation data found")
        return
    
    model_paths = {
        'CNN+LSTM': "cnn_lstm_audio_classification_model.h5",
        'CNN+BiLSTM': "cnn_lstm_eventdep_audio_classification_model.h5"
    }
    
    results = {}
    for name, path in model_paths.items():
        model = load_model(path)
        if model:
            X_val_expanded = np.expand_dims(X_val, axis=-1)
            y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(LABELS))
            loss, accuracy = model.evaluate(X_val_expanded, y_val_onehot, verbose=0)
            y_pred = np.argmax(model.predict(X_val_expanded, verbose=0), axis=1)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'true_labels': y_val
            }
    
    plot_comprehensive_results(results)
    
    # Print detailed reports
    for model_name, metrics in results.items():
        print(f"\n{model_name} Classification Report:")
        print(classification_report(metrics['true_labels'],
                                  metrics['predictions'],
                                  target_names=LABELS.values()))

if __name__ == "__main__":
    main()