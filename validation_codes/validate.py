
import numpy as np
import librosa
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from v_c_l import validate_cnn_lstm_model

SAMPLE_RATE = 22050
TARGET_DURATION = 3
TESTING_FOLDER = "validation_data"
FILE_EXTENSION = ".mp3"

N_MFCC = 13
N_MEL = 128
N_CHROMA = 12
MAX_FRAMES = 130

LABELS = {
    0: "Baseball Bat",
    1: "Bomb Explosion",
    2: "Hit and Run",
    3: "Kill Animals",
    4: "Lip Kissing",
    5: "None"
}

def load_and_pad_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_length = SAMPLE_RATE * TARGET_DURATION
        y = np.pad(y, (0, max(0, target_length - len(y))))[:target_length]
        return y, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def pad_or_truncate(feature, shape):
    if feature.shape[1] < shape[1]:
        pad_width = ((0, 0), (0, shape[1] - feature.shape[1]))
        feature = np.pad(feature, pad_width, mode='constant')
    return feature[:, :shape[1]]

def extract_features(file_path, model_type):
    y, sr = load_and_pad_audio(file_path)
    if y is None:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = pad_or_truncate(mfcc, (N_MFCC, MAX_FRAMES))
    
    if model_type == "only_mfcc":
        return mfcc

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
    mel_spec = pad_or_truncate(mel_spec, (N_MEL, MAX_FRAMES))
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    chroma = pad_or_truncate(chroma, (N_CHROMA, MAX_FRAMES))
    
    return np.vstack([mfcc, mel_spec, chroma])

def prepare_features(features, model_type):
    if features is None:
        return None
    return np.expand_dims(features, axis=(0, -1))

def evaluate_all_models(data_dir=TESTING_FOLDER):
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    model_paths = {
        "only_mfcc": "./models_created/only_mfcc_audio_classification_model.h5",
        "only_cnn": "./models_created/only_cnn_audio_classification_model.h5",
    }

    results = {}
    category_results = {model: {label: {"correct": 0, "total": 0} for label in LABELS.values()} 
                        for model in model_paths.keys()}
    
    file_paths, true_labels = [], []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(FILE_EXTENSION):
                file_path = os.path.join(root, file)
                try:
                    label = int(os.path.basename(root)) - 1
                    if label in LABELS:
                        file_paths.append(file_path)
                        true_labels.append(label)
                except:
                    continue

    if not file_paths:
        print("No valid files found for evaluation")
        return

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            results[model_name] = None
            continue
        
        print(f"\nEvaluating {model_name} model...")
        model = tf.keras.models.load_model(model_path)
        pred_labels = []

        for i, file_path in enumerate(file_paths):
            try:
                features = extract_features(file_path, model_name)
                prepared_features = prepare_features(features, model_name)
                if prepared_features is not None:
                    prediction = model.predict(prepared_features, verbose=0)
                    pred_label = np.argmax(prediction)
                    pred_labels.append(pred_label)

                    true_label = true_labels[i]
                    category_name = LABELS[true_label]
                    category_results[model_name][category_name]["total"] += 1
                    if pred_label == true_label:
                        category_results[model_name][category_name]["correct"] += 1
                else:
                    pred_labels.append(-1)
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                pred_labels.append(-1)

        valid_indices = [i for i, pred in enumerate(pred_labels) if pred != -1]
        filtered_true = [true_labels[i] for i in valid_indices]
        filtered_pred = [pred_labels[i] for i in valid_indices]

        if not filtered_true:
            accuracy = 0
        else:
            accuracy = accuracy_score(filtered_true, filtered_pred)
            print(f"Overall Accuracy: {accuracy:.2%}")
            print("Classification Report:")
            print(classification_report(filtered_true, filtered_pred, target_names=LABELS.values()))
        
        results[model_name] = accuracy

    # CNN+LSTM Evaluation
    print(f"\nEvaluating cnn_lstm model...")
    cnn_lstm_raw = validate_cnn_lstm_model("./models_created/cnn_lstm_eventdep_audio_classification_model.h5")

    cnn_lstm_formatted = {}
    correct_total = [0, 0]

    for category in LABELS.values():
        correct = cnn_lstm_raw[category]['correct']
        total = cnn_lstm_raw[category]['total']
        cnn_lstm_formatted[category] = {"correct": correct, "total": total}
        correct_total[0] += correct
        correct_total[1] += total

    category_results['cnn_lstm'] = cnn_lstm_formatted
    results['cnn_lstm'] = correct_total[0] / correct_total[1] if correct_total[1] > 0 else 0

    # Final Plots
    plot_category_results(category_results)
    plot_overall_results(results)
    return results, category_results

def plot_category_results(category_results):
    models = list(category_results.keys())
    categories = list(LABELS.values())
    accuracies = np.zeros((len(models), len(categories)))

    for i, model in enumerate(models):
        for j, category in enumerate(categories):
            total = category_results[model][category]["total"]
            correct = category_results[model][category]["correct"]
            accuracies[i, j] = (correct / total) * 100 if total > 0 else 0

    plt.figure(figsize=(14, 8))
    bar_width = 0.25
    index = np.arange(len(categories))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, model in enumerate(models):
        label = "CNN+LSTM" if model == "cnn_lstm" else model.replace('_', ' ').title()
        plt.bar(index + i * bar_width, accuracies[i], bar_width, label=label, color=colors[i])

    plt.xlabel('Audio Categories')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy by Category')
    plt.xticks(index + bar_width, categories, rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('category_accuracy_comparison.png', dpi=300)
    plt.show()

def plot_overall_results(results):
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("No valid results to plot")
        return

    model_names = ["CNN+LSTM" if k == "cnn_lstm" else k.replace('_', ' ').title() for k in valid_results.keys()]
    accuracies = [v * 100 for v in valid_results.values()]

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(model_names, accuracies, color=colors[:len(model_names)])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}%', ha='center', va='bottom')

    plt.title('Model Comparison - Overall Accuracy')
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('overall_accuracy_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("🔊 Audio Classification Model Evaluation")
    print("========================================")
    overall_results, category_results = evaluate_all_models()

    if overall_results:
        print("\nFinal Overall Results:")
        for model, accuracy in overall_results.items():
            label = "CNN+LSTM" if model == "cnn_lstm" else model.replace('_', ' ').title()
            print(f"{label}: {accuracy:.2%}")
