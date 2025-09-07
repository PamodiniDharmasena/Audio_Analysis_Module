import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from ultralytics import YOLO
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuration
IMG_SIZE = 64
SEQ_LENGTH = 20  # Number of frames per video
CATEGORIES = ['1', '2', '3', '4', '5', '6']
DATA_DIR = 'videos/'
YOLO_MODEL_PATH = 'yolov8n.pt'
MODEL_PATH = './visual_data_classification_model.keras'

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Preprocessing Functions
def denoise_frame(frame):
    """ Reduce noise using Gaussian blur. """
    return cv2.GaussianBlur(frame, (5, 5), 0)

def enhance_frame(frame):
    """ Enhance contrast using CLAHE. """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def compute_optical_flow(prev_frame, next_frame):
    """ Compute dense optical flow between two frames. """
    prev_frame = np.uint8(prev_frame) if prev_frame.dtype != np.uint8 else prev_frame
    next_frame = np.uint8(next_frame) if next_frame.dtype != np.uint8 else next_frame

    height, width = prev_frame.shape[:2]
    next_frame = cv2.resize(next_frame, (width, height))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def extract_features_with_yolo(frames):
    """ Extract object-based features using YOLO model. """
    features = []
    for frame in frames:
        results = yolo_model(frame)
        objects_detected = [d.name for d in results[0].boxes.data]

        violence_objects = ['gun', 'knife', 'drug']
        feature_vector = [1 if obj in objects_detected else 0 for obj in violence_objects]

        context_flag = 1 if 'gun' in objects_detected else 0  # Context-aware feature
        features.append(feature_vector + [context_flag])

    return np.array(features)

def preprocess_video(video_path):
    """ Extract, denoise, enhance, compute optical flow, and normalize frames. """
    frames = []
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    while len(frames) < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = denoise_frame(frame)  
        frame = enhance_frame(frame)  
        frame = frame / 255.0  

        # Compute optical flow
        if prev_frame is not None:
            flow = compute_optical_flow(prev_frame, frame)
        else:
            flow = np.zeros((IMG_SIZE, IMG_SIZE, 2))  

        frames.append((frame, flow))
        prev_frame = frame

    cap.release()
    return frames if len(frames) == SEQ_LENGTH else None

def load_videos():
    data, labels = [], []

    for category in CATEGORIES:
        folder_path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)

        for video in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video)
            frames = []
            cap = cv2.VideoCapture(video_path)

            while len(frames) < SEQ_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = frame / 255.0  
                frames.append(frame)

            cap.release()

            if len(frames) == SEQ_LENGTH:
                yolo_features = extract_features_with_yolo(frames)  

                yolo_features = np.repeat(yolo_features[:, np.newaxis, np.newaxis, :], IMG_SIZE, axis=1)
                yolo_features = np.repeat(yolo_features, IMG_SIZE, axis=2)  

                frames = np.stack(frames, axis=0)  

                combined_features = np.concatenate([frames, yolo_features], axis=-1)  

                data.append(combined_features)
                labels.append(class_num)

    return np.array(data), np.array(labels)

# Load and preprocess dataset
X, y = load_videos()
y = to_categorical(y, num_classes=len(CATEGORIES))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definition
def create_model():
    model = keras.Sequential()

    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 7)))  
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(layers.Flatten())

    model.add(layers.RepeatVector(SEQ_LENGTH))
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(CATEGORIES), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
model = create_model()
model.summary()

history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=8
)

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Plot Accuracy and Loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.show()

plot_training_history(history)

# Save Model
model.save(MODEL_PATH)

# Prediction Function
def predict_video(video_path):
    frames_with_flow = preprocess_video(video_path)
    if frames_with_flow:
        frames, flows = zip(*frames_with_flow)
        yolo_features = extract_features_with_yolo(frames)
        combined_features = np.concatenate([frames, flows, yolo_features], axis=-1)
        combined_features = np.expand_dims(combined_features, axis=0)
        prediction = model.predict(combined_features)
        class_idx = np.argmax(prediction)
        return CATEGORIES[class_idx]
    return "Not enough frames"

# Generate confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true_classes = np.argmax(y_test, axis=1)  # True class labels

# Compute confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(cm, CATEGORIES)

# Example Usage:
#  result = predict_video('newDataSet/videos/Verbal/sample.mp4')
# print(f'The video is classified as: {result}')
