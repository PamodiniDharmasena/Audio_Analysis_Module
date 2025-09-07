# # # import numpy as np
# # # import librosa
# # # import tensorflow as tf
# # # import os
# # # import matplotlib.pyplot as plt
# # # from sklearn.metrics import accuracy_score

# # # # Constants (must match training script)
# # # SAMPLE_RATE = 22050
# # # TARGET_DURATION = 3  # 3 seconds
# # # N_MFCC = 13
# # # N_MEL = 128
# # # N_CHROMA = 12
# # # SEQUENCE_LENGTH = 5  # Must match training

# # # # Label Mapping
# # # LABELS = {
# # #     0: "Baseball Bat",
# # #     1: "Bomb Explosion",
# # #     2: "Hit and Run",
# # #     3: "Kill Animals",
# # #     4: "Lip Kissing",
# # #     5: "None"
# # # }

# # # # Model paths
# # # MODEL_PATHS = {
# # #     "only_cnn": "only_cnn_audio_classification_model.h5",
# # #     "only_mfcc": "only_mfcc_audio_classification_model.h5",
# # #     "cnn_lstm_event_order": "cnn_lstm_audio_classification_model.h5"
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

# # # def extract_features_with_temporal_context(file_path):
# # #     """Extract features with temporal context (matches training)"""
# # #     y, sr = load_and_pad_audio(file_path)
    
# # #     # Split audio into SEQUENCE_LENGTH segments
# # #     segment_length = len(y) // SEQUENCE_LENGTH
# # #     features = []
    
# # #     for i in range(SEQUENCE_LENGTH):
# # #         segment = y[i*segment_length : (i+1)*segment_length]
        
# # #         # Extract features for each segment
# # #         mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
# # #         mfcc = pad_or_truncate(mfcc, (N_MFCC, 130//SEQUENCE_LENGTH))
        
# # #         mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL)
# # #         mel_spec = pad_or_truncate(mel_spec, (N_MEL, 130//SEQUENCE_LENGTH))
        
# # #         chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=N_CHROMA)
# # #         chroma = pad_or_truncate(chroma, (N_CHROMA, 130//SEQUENCE_LENGTH))
        
# # #         features.append(np.vstack([mfcc, mel_spec, chroma]))
    
# # #     return np.array(features)

# # # def pad_or_truncate(feature, shape):
# # #     """Ensure the feature matrix has a fixed shape."""
# # #     pad_width = [(0, max(0, shape[0] - feature.shape[0])), 
# # #                 (0, max(0, shape[1] - feature.shape[1]))]
# # #     feature = np.pad(feature, pad_width, mode='constant')
# # #     return feature[:, :shape[1]]  # Truncate if needed

# # # def evaluate_model(model, data_dir):
# # #     """Evaluate a model on the validation dataset"""
# # #     true_labels = []
# # #     pred_labels = []
    
# # #     for label_idx, label_name in LABELS.items():
# # #         label_dir = os.path.join(data_dir, str(label_idx + 1))  # Labels are 1-6 in folder names
# # #         if not os.path.exists(label_dir):
# # #             continue
            
# # #         for audio_file in os.listdir(label_dir):
# # #             if not audio_file.endswith(('.wav', '.mp3')):
# # #                 continue
                
# # #             file_path = os.path.join(label_dir, audio_file)
# # #             features = extract_features_with_temporal_context(file_path)
            
# # #             if features is None:
# # #                 continue
                
# # #             # Prepare features for prediction
# # #             features = np.expand_dims(features, axis=-1)  # Add channel dimension
# # #             features = np.expand_dims(features, axis=0)   # Add batch dimension
            
# # #             # Get prediction
# # #             prediction = model.predict(features)
# # #             predicted_label = np.argmax(prediction)
            
# # #             true_labels.append(label_idx)
# # #             pred_labels.append(predicted_label)
    
# # #     accuracy = accuracy_score(true_labels, pred_labels)
# # #     return accuracy

# # # def evaluate_all_models(data_dir):
# # #     """Evaluate all three models and plot results"""
# # #     accuracies = {}
    
# # #     for model_name, model_path in MODEL_PATHS.items():
# # #         try:
# # #             print(f"Loading {model_name}...")
# # #             model = tf.keras.models.load_model(model_path)
# # #             accuracy = evaluate_model(model, data_dir)
# # #             accuracies[model_name] = accuracy
# # #             print(f"{model_name} validation accuracy: {accuracy:.2%}")
# # #         except Exception as e:
# # #             print(f"Error evaluating {model_name}: {str(e)}")
# # #             accuracies[model_name] = 0
    
# # #     # Plot results
# # #     plt.figure(figsize=(10, 6))
# # #     bars = plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
# # #     plt.title('Model Validation Accuracy Comparison')
# # #     plt.xlabel('Model')
# # #     plt.ylabel('Accuracy')
# # #     plt.ylim(0, 1)
    
# # #     # Add accuracy values on top of bars
# # #     for bar in bars:
# # #         height = bar.get_height()
# # #         plt.text(bar.get_x() + bar.get_width()/2., height,
# # #                  f'{height:.2%}', ha='center', va='bottom')
    
# # #     plt.tight_layout()
# # #     plt.savefig('model_comparison.png')
# # #     plt.show()
    
# # #     return accuracies

# # # if __name__ == "__main__":
# # #     validation_data_dir = "./validation _data"  # Path to your validation data folder
# # #     if not os.path.exists(validation_data_dir):
# # #         print(f"Error: Validation data directory not found at {validation_data_dir}")
# # #     else:
# # #         accuracies = evaluate_all_models(validation_data_dir)
# # #         print("\nFinal Validation Accuracies:")
# # #         for model, acc in accuracies.items():
# # #             print(f"{model}: {acc:.2%}")



# # # import numpy as np
# # # import librosa
# # # import tensorflow as tf
# # # import os
# # # import matplotlib.pyplot as plt
# # # from sklearn.metrics import accuracy_score

# # # # Constants (must match training script)
# # # SAMPLE_RATE = 22050
# # # TARGET_DURATION = 3  # 3 seconds
# # # N_MFCC = 13
# # # N_MEL = 128
# # # N_CHROMA = 12
# # # SEQUENCE_LENGTH = 5  # Must match training

# # # # Label Mapping
# # # LABELS = {
# # #     0: "Baseball Bat",
# # #     1: "Bomb Explosion",
# # #     2: "Hit and Run",
# # #     3: "Kill Animals",
# # #     4: "Lip Kissing",
# # #     5: "None"
# # # }

# # # # Model paths
# # # MODEL_PATHS = {
# # #     "only_cnn": "only_cnn_audio_classification_model.h5",
# # #     "only_mfcc": "only_mfcc_audio_classification_model.h5",
# # #     "cnn_lstm_event_order": "cnn_lstm_audio_classification_model.h5"
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

# # # def extract_features_for_model(file_path, model_name):
# # #     """Extract features specific to each model type"""
# # #     y, sr = load_and_pad_audio(file_path)
    
# # #     if model_name == "only_cnn":
# # #         # CNN expects single 2D feature map
# # #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# # #         mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
# # #         chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
# # #         features = np.vstack([mfcc, mel_spec, chroma])
# # #         features = pad_or_truncate(features, (153, 130))  # Adjust dimensions as needed
# # #         return np.expand_dims(features, axis=-1)  # Add channel dimension
        
# # #     elif model_name == "only_mfcc":
# # #         # MFCC-only model expects different features
# # #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# # #         features = pad_or_truncate(mfcc, (N_MFCC, 130))
# # #         return np.expand_dims(features, axis=-1)
        
# # #     elif model_name == "cnn_lstm_event_order":
# # #         # CNN-LSTM expects temporal features
# # #         segment_length = len(y) // SEQUENCE_LENGTH
# # #         features = []
        
# # #         for i in range(SEQUENCE_LENGTH):
# # #             segment = y[i*segment_length : (i+1)*segment_length]
# # #             mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
# # #             mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL)
# # #             chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=N_CHROMA)
# # #             segment_features = np.vstack([mfcc, mel_spec, chroma])
# # #             segment_features = pad_or_truncate(segment_features, (153, 26))  # Adjust dimensions
# # #             features.append(segment_features)
            
# # #         features = np.array(features)
# # #         return np.expand_dims(features, axis=-1)  # Add channel dimension

# # # def pad_or_truncate(feature, shape):
# # #     """Ensure the feature matrix has a fixed shape."""
# # #     pad_width = [(0, max(0, shape[0] - feature.shape[0])), 
# # #                 (0, max(0, shape[1] - feature.shape[1]))]
# # #     feature = np.pad(feature, pad_width, mode='constant')
# # #     return feature[:, :shape[1]]  # Truncate if needed

# # # def evaluate_model(model, model_name, data_dir):
# # #     """Evaluate a model on the validation dataset"""
# # #     true_labels = []
# # #     pred_labels = []
    
# # #     for label_idx, label_name in LABELS.items():
# # #         label_dir = os.path.join(data_dir, str(label_idx + 1))  # Labels are 1-6 in folder names
# # #         if not os.path.exists(label_dir):
# # #             continue
            
# # #         for audio_file in os.listdir(label_dir):
# # #             if not audio_file.endswith(('.wav', '.mp3')):
# # #                 continue
                
# # #             file_path = os.path.join(label_dir, audio_file)
# # #             features = extract_features_for_model(file_path, model_name)
            
# # #             if features is None:
# # #                 continue
                
# # #             # Prepare features for prediction
# # #             if model_name == "cnn_lstm_event_order":
# # #                 features = np.expand_dims(features, axis=0)  # Add batch dimension for LSTM
# # #             else:
# # #                 features = np.expand_dims(features, axis=0)  # Add batch dimension for CNN
                
# # #             # Get prediction
# # #             try:
# # #                 prediction = model.predict(features)
# # #                 predicted_label = np.argmax(prediction)
# # #                 true_labels.append(label_idx)
# # #                 pred_labels.append(predicted_label)
# # #             except Exception as e:
# # #                 print(f"Error predicting with {model_name}: {str(e)}")
# # #                 continue
    
# # #     if len(true_labels) == 0:
# # #         return 0.0
        
# # #     accuracy = accuracy_score(true_labels, pred_labels)
# # #     return accuracy

# # # def evaluate_all_models(data_dir):
# # #     """Evaluate all three models and plot results"""
# # #     accuracies = {}
    
# # #     for model_name, model_path in MODEL_PATHS.items():
# # #         try:
# # #             print(f"Loading {model_name}...")
# # #             model = tf.keras.models.load_model(model_path)
# # #             accuracy = evaluate_model(model, model_name, data_dir)
# # #             accuracies[model_name] = accuracy
# # #             print(f"{model_name} validation accuracy: {accuracy:.2%}")
# # #         except Exception as e:
# # #             print(f"Error evaluating {model_name}: {str(e)}")
# # #             accuracies[model_name] = 0
    
# # #     # Plot results
# # #     plt.figure(figsize=(10, 6))
# # #     bars = plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
# # #     plt.title('Model Validation Accuracy Comparison')
# # #     plt.xlabel('Model')
# # #     plt.ylabel('Accuracy')
# # #     plt.ylim(0, 1)
    
# # #     # Add accuracy values on top of bars
# # #     for bar in bars:
# # #         height = bar.get_height()
# # #         plt.text(bar.get_x() + bar.get_width()/2., height,
# # #                  f'{height:.2%}', ha='center', va='bottom')
    
# # #     plt.tight_layout()
# # #     plt.savefig('model_comparison.png')
# # #     plt.show()
    
# # #     return accuracies

# # # if __name__ == "__main__":
# # #     validation_data_dir = "./validation_data"  # Path to your validation data folder
# # #     if not os.path.exists(validation_data_dir):
# # #         print(f"Error: Validation data directory not found at {validation_data_dir}")
# # #     else:
# # #         accuracies = evaluate_all_models(validation_data_dir)
# # #         print("\nFinal Validation Accuracies:")
# # #         for model, acc in accuracies.items():
# # #             print(f"{model}: {acc:.2%}")









# # # import numpy as np
# # # import librosa
# # # import tensorflow as tf
# # # import os
# # # import matplotlib.pyplot as plt
# # # from sklearn.metrics import accuracy_score

# # # # Constants
# # # SAMPLE_RATE = 22050
# # # TARGET_DURATION = 3  # seconds
# # # N_MFCC = 13
# # # N_MEL = 128
# # # N_CHROMA = 12
# # # SEQUENCE_LENGTH = 5

# # # # Label Mapping
# # # LABELS = {
# # #     0: "Baseball Bat",
# # #     1: "Bomb Explosion",
# # #     2: "Hit and Run",
# # #     3: "Kill Animals",
# # #     4: "Lip Kissing",
# # #     5: "None"
# # # }

# # # # Model paths
# # # MODEL_PATHS = {
# # #     "only_cnn": "only_cnn_audio_classification_model.h5",
# # #     "only_mfcc": "only_mfcc_audio_classification_model.h5",
# # #     "cnn_lstm_event_order": "cnn_lstm_audio_classification_model.h5"
# # # }


# # # def load_and_pad_audio(file_path):
# # #     y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
# # #     target_length = SAMPLE_RATE * TARGET_DURATION
# # #     if len(y) < target_length:
# # #         y = np.pad(y, (0, target_length - len(y)))
# # #     else:
# # #         y = y[:target_length]
# # #     return y, sr


# # # def get_model_input_shape(model):
# # #     """Get the expected input shape of a model"""
# # #     return model.input_shape[1:]  # Exclude batch dimension


# # # def pad_or_truncate(feature, shape):
# # #     pad_width = [
# # #         (0, max(0, shape[0] - feature.shape[0])),
# # #         (0, max(0, shape[1] - feature.shape[1]))
# # #     ]
# # #     feature = np.pad(feature, pad_width, mode='constant')
# # #     return feature[:shape[0], :shape[1]]


# # # def extract_features_for_model(file_path, model_name, model):
# # #     y, sr = load_and_pad_audio(file_path)
# # #     input_shape = get_model_input_shape(model)

# # #     if model_name == "only_cnn":
# # #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# # #         mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
# # #         chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
# # #         features = np.vstack([mfcc, mel, chroma])
# # #         features = pad_or_truncate(features, (input_shape[0], input_shape[1]))
# # #         return np.expand_dims(features, axis=-1)

# # #     elif model_name == "only_mfcc":
# # #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# # #         features = pad_or_truncate(mfcc, (input_shape[0], input_shape[1]))
# # #         return np.expand_dims(features, axis=-1)

# # #     elif model_name == "cnn_lstm_event_order":
# # #         segment_length = len(y) // SEQUENCE_LENGTH
# # #         sequence = []
# # #         for i in range(SEQUENCE_LENGTH):
# # #             segment = y[i * segment_length: (i + 1) * segment_length]
# # #             mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
# # #             mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL)
# # #             chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=N_CHROMA)
# # #             features = np.vstack([mfcc, mel, chroma])
# # #             features = pad_or_truncate(features, (input_shape[1], input_shape[2]))  # skip timesteps
# # #             sequence.append(features)
# # #         sequence = np.array(sequence)
# # #         return np.expand_dims(sequence, axis=-1)

# # #     return None


# # # def evaluate_model(model, model_name, data_dir):
# # #     true_labels = []
# # #     pred_labels = []

# # #     for label_idx, label_name in LABELS.items():
# # #         label_dir = os.path.join(data_dir, str(label_idx + 1))  # Folders are 1 to 6
# # #         if not os.path.exists(label_dir):
# # #             continue

# # #         for audio_file in os.listdir(label_dir):
# # #             if not audio_file.lower().endswith(('.wav', '.mp3')):
# # #                 continue

# # #             file_path = os.path.join(label_dir, audio_file)
# # #             try:
# # #                 features = extract_features_for_model(file_path, model_name, model)
# # #                 if features is None:
# # #                     continue
# # #                 features = np.expand_dims(features, axis=0)  # Batch dimension
# # #                 prediction = model.predict(features, verbose=0)
# # #                 predicted_label = np.argmax(prediction)
# # #                 true_labels.append(label_idx)
# # #                 pred_labels.append(predicted_label)
# # #             except Exception as e:
# # #                 print(f"Error predicting {audio_file} in {model_name}: {str(e)}")

# # #     if len(true_labels) == 0:
# # #         return 0.0

# # #     accuracy = accuracy_score(true_labels, pred_labels)
# # #     return accuracy


# # # def evaluate_all_models(data_dir):
# # #     accuracies = {}

# # #     for model_name, model_path in MODEL_PATHS.items():
# # #         try:
# # #             print(f"\nLoading {model_name}...")
# # #             model = tf.keras.models.load_model(model_path)
# # #             input_shape = get_model_input_shape(model)
# # #             print(f"{model_name} input shape: {input_shape}")
# # #             accuracy = evaluate_model(model, model_name, data_dir)
# # #             accuracies[model_name] = accuracy
# # #             print(f"{model_name} validation accuracy: {accuracy:.2%}")
# # #         except Exception as e:
# # #             print(f"Error evaluating {model_name}: {str(e)}")
# # #             accuracies[model_name] = 0.0

# # #     # Plotting
# # #     plt.figure(figsize=(10, 6))
# # #     bars = plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
# # #     plt.title('Model Validation Accuracy Comparison')
# # #     plt.xlabel('Model')
# # #     plt.ylabel('Accuracy')
# # #     plt.ylim(0, 1)

# # #     for bar in bars:
# # #         height = bar.get_height()
# # #         plt.text(bar.get_x() + bar.get_width() / 2., height,
# # #                  f'{height:.2%}', ha='center', va='bottom')

# # #     plt.tight_layout()
# # #     plt.savefig('model_comparison.png')
# # #     plt.show()

# # #     return accuracies


# # # if __name__ == "__main__":
# # #     validation_data_dir = "./validation_data"
# # #     if not os.path.exists(validation_data_dir):
# # #         print(f"Error: Validation data directory not found at {validation_data_dir}")
# # #     else:
# # #         accuracies = evaluate_all_models(validation_data_dir)
# # #         print("\nFinal Validation Accuracies:")
# # #         for model, acc in accuracies.items():
# # #             print(f"{model}: {acc:.2%}")




# # import numpy as np
# # import librosa
# # import tensorflow as tf
# # import os
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import accuracy_score

# # # Constants
# # SAMPLE_RATE = 22050
# # TARGET_DURATION = 3  # seconds
# # N_MFCC = 13
# # N_MEL = 128
# # N_CHROMA = 12
# # MAX_FRAMES = 130
# # SEQUENCE_LENGTH = 5

# # # Label Mapping
# # LABELS = {
# #     0: "Baseball Bat",
# #     1: "Bomb Explosion",
# #     2: "Hit and Run",
# #     3: "Kill Animals",
# #     4: "Lip Kissing",
# #     5: "None"
# # }

# # # Model paths
# # MODEL_PATHS = {
# #     "only_cnn": "only_cnn_audio_classification_model.h5",
# #     "only_mfcc": "only_mfcc_audio_classification_model.h5",
# #     "cnn_lstm_event_order": "cnn_lstm_audio_classification_model.h5"
# # }

# # def load_and_pad_audio(file_path):
# #     """Load and pad/truncate audio to fixed duration"""
# #     y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
# #     target_length = SAMPLE_RATE * TARGET_DURATION
# #     if len(y) < target_length:
# #         y = np.pad(y, (0, target_length - len(y)))
# #     else:
# #         y = y[:target_length]
# #     return y, sr

# # def extract_cnn_features(file_path):
# #     """Feature extraction for CNN model"""
# #     y, sr = load_and_pad_audio(file_path)
# #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# #     mfcc = pad_or_truncate(mfcc, (N_MFCC, MAX_FRAMES))
# #     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
# #     mel_spec = pad_or_truncate(mel_spec, (N_MEL, MAX_FRAMES))
# #     chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
# #     chroma = pad_or_truncate(chroma, (N_CHROMA, MAX_FRAMES))
# #     features = np.vstack([mfcc, mel_spec, chroma])
# #     return np.expand_dims(features, axis=-1)  # (153, 130, 1)

# # def extract_mfcc_features(file_path):
# #     """Feature extraction for MFCC-only model"""
# #     y, sr = load_and_pad_audio(file_path)
# #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# #     if mfcc.shape[1] > MAX_FRAMES:
# #         mfcc = mfcc[:, :MAX_FRAMES]
# #     else:
# #         pad_width = MAX_FRAMES - mfcc.shape[1]
# #         mfcc = np.pad(mfcc, ((0,0), (0,pad_width)), mode='constant')
# #     return np.expand_dims(mfcc, axis=-1)  # (13, 130, 1)

# # def extract_cnn_lstm_features(file_path):
# #     """Feature extraction for CNN-LSTM model - CRITICAL FIX"""
# #     y, sr = load_and_pad_audio(file_path)
    
# #     # Extract full features first (153, 130)
# #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
# #     mfcc = pad_or_truncate(mfcc, (N_MFCC, MAX_FRAMES))
    
# #     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
# #     mel_spec = pad_or_truncate(mel_spec, (N_MEL, MAX_FRAMES))
    
# #     chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
# #     chroma = pad_or_truncate(chroma, (N_CHROMA, MAX_FRAMES))
    
# #     full_features = np.vstack([mfcc, mel_spec, chroma])  # (153, 130)
    
# #     # Split into segments along time axis (width)
# #     segment_width = full_features.shape[1] // SEQUENCE_LENGTH
# #     segments = []
# #     for i in range(SEQUENCE_LENGTH):
# #         segment = full_features[:, i*segment_width:(i+1)*segment_width]
# #         segments.append(segment)
    
# #     # Stack segments and add channel dimension
# #     features = np.array(segments)  # (5, 153, 26)
# #     features = np.expand_dims(features, axis=-1)  # (5, 153, 26, 1)
    
# #     return features

# # def pad_or_truncate(feature, shape):
# #     """Ensure feature matrix has fixed shape"""
# #     pad_width = [(0, max(0, shape[0] - feature.shape[0])), 
# #                 (0, max(0, shape[1] - feature.shape[1]))]
# #     feature = np.pad(feature, pad_width, mode='constant')
# #     return feature[:, :shape[1]]

# # def evaluate_cnn_model(data_dir):
# #     """Evaluate CNN model"""
# #     model = tf.keras.models.load_model(MODEL_PATHS["only_cnn"])
# #     true_labels = []
# #     pred_labels = []
    
# #     for label_idx, label_name in LABELS.items():
# #         label_dir = os.path.join(data_dir, str(label_idx + 1))
# #         if not os.path.exists(label_dir):
# #             continue
            
# #         for audio_file in os.listdir(label_dir):
# #             if not audio_file.endswith(('.wav', '.mp3')):
# #                 continue
                
# #             file_path = os.path.join(label_dir, audio_file)
# #             features = extract_cnn_features(file_path)
# #             features = np.expand_dims(features, axis=0)  # Add batch dim (1, 153, 130, 1)
            
# #             try:
# #                 prediction = model.predict(features, verbose=0)
# #                 pred_labels.append(np.argmax(prediction))
# #                 true_labels.append(label_idx)
# #             except Exception as e:
# #                 print(f"Error predicting {audio_file}: {str(e)}")
# #                 continue
    
# #     accuracy = accuracy_score(true_labels, pred_labels) if true_labels else 0
# #     print(f"CNN model validation accuracy: {accuracy:.2%}")
# #     return accuracy

# # def evaluate_mfcc_model(data_dir):
# #     """Evaluate MFCC model"""
# #     model = tf.keras.models.load_model(MODEL_PATHS["only_mfcc"])
# #     true_labels = []
# #     pred_labels = []
    
# #     for label_idx, label_name in LABELS.items():
# #         label_dir = os.path.join(data_dir, str(label_idx + 1))
# #         if not os.path.exists(label_dir):
# #             continue
            
# #         for audio_file in os.listdir(label_dir):
# #             if not audio_file.endswith(('.wav', '.mp3')):
# #                 continue
                
# #             file_path = os.path.join(label_dir, audio_file)
# #             features = extract_mfcc_features(file_path)
# #             features = np.expand_dims(features, axis=0)  # Add batch dim (1, 13, 130, 1)
            
# #             try:
# #                 prediction = model.predict(features, verbose=0)
# #                 pred_labels.append(np.argmax(prediction))
# #                 true_labels.append(label_idx)
# #             except Exception as e:
# #                 print(f"Error predicting {audio_file}: {str(e)}")
# #                 continue
    
# #     accuracy = accuracy_score(true_labels, pred_labels) if true_labels else 0
# #     print(f"MFCC model validation accuracy: {accuracy:.2%}")
# #     return accuracy

# # def evaluate_cnn_lstm_model(data_dir):
# #     """Evaluate CNN-LSTM model"""
# #     model = tf.keras.models.load_model(MODEL_PATHS["cnn_lstm_event_order"])
# #     true_labels = []
# #     pred_labels = []
    
# #     for label_idx, label_name in LABELS.items():
# #         label_dir = os.path.join(data_dir, str(label_idx + 1))
# #         if not os.path.exists(label_dir):
# #             continue
            
# #         for audio_file in os.listdir(label_dir):
# #             if not audio_file.endswith(('.wav', '.mp3')):
# #                 continue
                
# #             file_path = os.path.join(label_dir, audio_file)
# #             features = extract_cnn_lstm_features(file_path)
# #             features = np.expand_dims(features, axis=0)  # Add batch dim (1, 5, 153, 26, 1)
            
# #             try:
# #                 # Reshape features to match model input shape
# #                 # The model expects (batch_size, timesteps, height, width, channels)
# #                 prediction = model.predict(features, verbose=0)
# #                 pred_labels.append(np.argmax(prediction))
# #                 true_labels.append(label_idx)
# #             except Exception as e:
# #                 print(f"Error predicting {audio_file}: {str(e)}")
# #                 continue
    
# #     accuracy = accuracy_score(true_labels, pred_labels) if true_labels else 0
# #     print(f"CNN-LSTM model validation accuracy: {accuracy:.2%}")
# #     return accuracy

# # def plot_results(accuracies):
# #     """Plot comparison graph"""
# #     plt.figure(figsize=(10, 6))
# #     bars = plt.bar(accuracies.keys(), accuracies.values(), 
# #                   color=['blue', 'green', 'red'])
# #     plt.title('Model Validation Accuracy Comparison')
# #     plt.xlabel('Model')
# #     plt.ylabel('Accuracy')
# #     plt.ylim(0, 1)
    
# #     for bar in bars:
# #         height = bar.get_height()
# #         plt.text(bar.get_x() + bar.get_width()/2., height,
# #                 f'{height:.2%}', ha='center', va='bottom')
    
# #     plt.tight_layout()
# #     plt.savefig('model_comparison.png')
# #     plt.show()

# # if __name__ == "__main__":
# #     validation_data_dir = "./validation_data"
# #     if not os.path.exists(validation_data_dir):
# #         print(f"Error: Validation data directory not found at {validation_data_dir}")
# #     else:
# #         # Evaluate each model separately
# #         print("\nEvaluating CNN model...")
# #         cnn_acc = evaluate_cnn_model(validation_data_dir)
        
# #         print("\nEvaluating MFCC model...")
# #         mfcc_acc = evaluate_mfcc_model(validation_data_dir)
        
# #         print("\nEvaluating CNN-LSTM model...")
# #         cnn_lstm_acc = evaluate_cnn_lstm_model(validation_data_dir)
        
# #         # Collect results
# #         accuracies = {
# #             "only_cnn": cnn_acc,
# #             "only_mfcc": mfcc_acc,
# #             "cnn_lstm_event_order": cnn_lstm_acc
# #         }
        
# #         # Plot comparison
# #         print("\nFinal Validation Accuracies:")
# #         for model, acc in accuracies.items():
# #             print(f"{model}: {acc:.2%}")
        
# #         plot_results(accuracies)


# import numpy as np
# import librosa
# import tensorflow as tf
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score

# # Constants
# SAMPLE_RATE = 22050
# TARGET_DURATION = 3  # seconds
# N_MFCC = 13
# N_MEL = 128
# N_CHROMA = 12
# MAX_FRAMES = 130
# SEQUENCE_LENGTH = 5

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
#     """Load and pad/truncate audio to fixed duration"""
#     y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
#     target_length = SAMPLE_RATE * TARGET_DURATION
#     if len(y) < target_length:
#         y = np.pad(y, (0, target_length - len(y)))
#     else:
#         y = y[:target_length]
#     return y, sr

# def extract_cnn_lstm_features(file_path):
#     """Feature extraction for CNN-LSTM model"""
#     y, sr = load_and_pad_audio(file_path)
    
#     # Extract full features first (153, 130)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
#     mfcc = pad_or_truncate(mfcc, (N_MFCC, MAX_FRAMES))
    
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
#     mel_spec = pad_or_truncate(mel_spec, (N_MEL, MAX_FRAMES))
    
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
#     chroma = pad_or_truncate(chroma, (N_CHROMA, MAX_FRAMES))
    
#     full_features = np.vstack([mfcc, mel_spec, chroma])  # (153, 130)
    
#     # Split into segments along time axis (width)
#     segment_width = full_features.shape[1] // SEQUENCE_LENGTH
#     segments = []
#     for i in range(SEQUENCE_LENGTH):
#         segment = full_features[:, i*segment_width:(i+1)*segment_width]
#         segments.append(segment)
    
#     # Stack segments and add channel dimension
#     features = np.array(segments)  # (5, 153, 26)
#     features = np.expand_dims(features, axis=-1)  # (5, 153, 26, 1)
    
#     return features

# def pad_or_truncate(feature, shape):
#     """Ensure feature matrix has fixed shape"""
#     pad_width = [(0, max(0, shape[0] - feature.shape[0])), 
#                 (0, max(0, shape[1] - feature.shape[1]))]
#     feature = np.pad(feature, pad_width, mode='constant')
#     return feature[:, :shape[1]]

# def evaluate_cnn_lstm_model(data_dir, model_path):
#     """Evaluate CNN-LSTM model"""
#     model = tf.keras.models.load_model(model_path)
#     true_labels = []
#     pred_labels = []
    
#     for label_idx, label_name in LABELS.items():
#         label_dir = os.path.join(data_dir, str(label_idx + 1))
#         if not os.path.exists(label_dir):
#             continue
            
#         for audio_file in os.listdir(label_dir):
#             if not audio_file.endswith(('.wav', '.mp3')):
#                 continue
                
#             file_path = os.path.join(label_dir, audio_file)
#             features = extract_cnn_lstm_features(file_path)
#             features = np.expand_dims(features, axis=0)  # Add batch dim (1, 5, 153, 26, 1)
            
#             try:
#                 prediction = model.predict(features, verbose=0)
#                 pred_labels.append(np.argmax(prediction))
#                 true_labels.append(label_idx)
#             except Exception as e:
#                 print(f"Error predicting {audio_file}: {str(e)}")
#                 continue
    
#     accuracy = accuracy_score(true_labels, pred_labels) if true_labels else 0
#     print(f"CNN-LSTM model validation accuracy: {accuracy:.2%}")
#     return accuracy

# if __name__ == "__main__":
#     validation_data_dir = "./validation_data"
#     model_path = "cnn_lstm_audio_classification_model.h5"
    
#     if not os.path.exists(validation_data_dir):
#         print(f"Error: Validation data directory not found at {validation_data_dir}")
#     elif not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#     else:
#         print("Evaluating CNN-LSTM model...")
#         accuracy = evaluate_cnn_lstm_model(validation_data_dir, model_path)
#         print(f"\nFinal Validation Accuracy: {accuracy:.2%}")

















import numpy as np
import librosa
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
SAMPLE_RATE = 22050
TARGET_DURATION = 3  # seconds
TESTING_FOLDER = "validation_data"
FILE_EXTENSION = ".mp3"

# Feature Dimensions
N_MFCC = 13
N_MEL = 128
N_CHROMA = 12
MAX_FRAMES = 130
SEQUENCE_LENGTH = 5  # For CNN-LSTM model

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
    """Load and pad/truncate audio to fixed duration"""
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

def pad_or_truncate(feature, shape):
    """Ensure feature matrix has fixed shape"""
    if feature.shape[1] < shape[1]:
        pad_width = ((0, 0), (0, shape[1] - feature.shape[1]))
        feature = np.pad(feature, pad_width, mode='constant')
    return feature[:, :shape[1]]

def extract_features(file_path, model_type):
    """Feature extraction for all model types"""
    y, sr = load_and_pad_audio(file_path)
    if y is None:
        return None

    # Common MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = pad_or_truncate(mfcc, (N_MFCC, MAX_FRAMES))
    
    if model_type == "only_mfcc":
        return mfcc
    
    # Additional features for CNN and CNN-LSTM models
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
    mel_spec = pad_or_truncate(mel_spec, (N_MEL, MAX_FRAMES))
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    chroma = pad_or_truncate(chroma, (N_CHROMA, MAX_FRAMES))
    
    full_features = np.vstack([mfcc, mel_spec, chroma])  # (153, 130)
    
    if model_type == "only_cnn":
        return full_features
    elif model_type == "cnn_lstm":
        # Split into segments for CNN-LSTM
        segment_width = full_features.shape[1] // SEQUENCE_LENGTH
        segments = []
        for i in range(SEQUENCE_LENGTH):
            segment = full_features[:, i*segment_width:(i+1)*segment_width]
            segments.append(segment)
        return np.array(segments)  # (5, 153, 26)
    
    return None

def prepare_features(features, model_type):
    """Prepare features for model prediction"""
    if features is None:
        return None
        
    if model_type == "only_mfcc":
        return np.expand_dims(features, axis=(0, -1))  # (1, 13, 130, 1)
    elif model_type == "only_cnn":
        return np.expand_dims(features, axis=(0, -1))  # (1, 153, 130, 1)
    elif model_type == "cnn_lstm":
        return np.expand_dims(features, axis=(0, -1))  # (1, 5, 153, 26, 1)
    return None

def evaluate_all_models(data_dir=TESTING_FOLDER):
    """Evaluate all three models and display comparative results"""
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    model_paths = {
        "only_mfcc": "only_mfcc_audio_classification_model.h5",
        "only_cnn": "only_cnn_audio_classification_model.h5",
        "cnn_lstm": "cnn_lstm_audio_classification_model.h5"
    }
    
    results = {}
    
    # First collect all files and true labels
    file_paths = []
    true_labels = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(FILE_EXTENSION):
                file_path = os.path.join(root, file)
                try:
                    label = int(os.path.basename(root)) - 1  # Assuming folders are 1-6
                    if label in LABELS:
                        file_paths.append(file_path)
                        true_labels.append(label)
                except:
                    continue
    
    if not file_paths:
        print("No valid files found for evaluation")
        return
    
    # Evaluate each model
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            results[model_name] = None
            continue
        
        print(f"\nEvaluating {model_name} model...")
        model = tf.keras.models.load_model(model_path)
        pred_labels = []
        
        for file_path in file_paths:
            try:
                features = extract_features(file_path, model_name)
                prepared_features = prepare_features(features, model_name)
                
                if prepared_features is not None:
                    prediction = model.predict(prepared_features, verbose=0)
                    pred_label = np.argmax(prediction)
                    pred_labels.append(pred_label)
                else:
                    pred_labels.append(-1)  # Mark as invalid
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                pred_labels.append(-1)
                continue
        
        # Filter out invalid predictions
        valid_indices = [i for i, pred in enumerate(pred_labels) if pred != -1]
        filtered_true = [true_labels[i] for i in valid_indices]
        filtered_pred = [pred_labels[i] for i in valid_indices]
        
        if not filtered_true:
            accuracy = 0
            print("No valid predictions for this model")
        else:
            accuracy = accuracy_score(filtered_true, filtered_pred)
            print(f"Accuracy: {accuracy:.2%}")
            print("Classification Report:")
            print(classification_report(filtered_true, filtered_pred, target_names=LABELS.values()))
        
        results[model_name] = accuracy
    
    # Plot comparative results
    plot_results(results)
    
    return results

def plot_results(results):
    """Plot the comparative results as a bar graph"""
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    model_names = list(valid_results.keys())
    accuracies = [v * 100 for v in valid_results.values()]  # Convert to percentage
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.title('Model Comparison - Validation Accuracy', fontsize=14)
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🔊 Audio Classification Model Evaluation")
    print("========================================")
    print("Evaluating all three models...\n")
    
    results = evaluate_all_models()
    
    if results:
        print("\nFinal Results:")
        for model, accuracy in results.items():
            if accuracy is not None:
                print(f"{model}: {accuracy:.2%}")