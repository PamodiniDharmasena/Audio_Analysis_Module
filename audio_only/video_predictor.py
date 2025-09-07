import os
import cv2
import numpy as np
from tensorflow import keras
from ultralytics import YOLO

# Configuration
IMG_SIZE = 64
SEQ_LENGTH = 20
CATEGORIES = ['1', '2', '3', '4', '5', '6']
YOLO_MODEL_PATH = 'yolov8n.pt'
MODEL_PATH = './visual_data_classification_model.keras'

# Load trained model
model = keras.models.load_model(MODEL_PATH)

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# --- Utility Functions (from training script) ---
def denoise_frame(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def compute_optical_flow(prev_frame, next_frame):
    prev_frame = np.uint8(prev_frame)
    next_frame = np.uint8(next_frame)
    height, width = prev_frame.shape[:2]
    next_frame = cv2.resize(next_frame, (width, height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def extract_features_with_yolo(frames):
    features = []
    for frame in frames:
        results = yolo_model(frame)
        objects_detected = [d.name for d in results[0].boxes.data]
        violence_objects = ['gun', 'knife', 'drug']
        feature_vector = [1 if obj in objects_detected else 0 for obj in violence_objects]
        context_flag = 1 if 'gun' in objects_detected else 0
        features.append(feature_vector + [context_flag])
    return np.array(features)

def preprocess_video(video_path):
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
        if prev_frame is not None:
            flow = compute_optical_flow(prev_frame, frame)
        else:
            flow = np.zeros((IMG_SIZE, IMG_SIZE, 2))
        frames.append((frame, flow))
        prev_frame = frame

    cap.release()
    return frames if len(frames) == SEQ_LENGTH else None

# --- Prediction Function ---
# def predict_video(video_path):
#     frames_with_flow = preprocess_video(video_path)
#     if frames_with_flow:
#         frames, _ = zip(*frames_with_flow)  # We ignore the flow component

#         yolo_features = extract_features_with_yolo(frames)

#         yolo_features = np.repeat(yolo_features[:, np.newaxis, np.newaxis, :], IMG_SIZE, axis=1)
#         yolo_features = np.repeat(yolo_features, IMG_SIZE, axis=2)

#         frames = np.stack(frames, axis=0)
#         combined_features = np.concatenate([frames, yolo_features], axis=-1)

#         combined_features = np.expand_dims(combined_features, axis=0)

#         prediction = model.predict(combined_features)
#         class_idx = np.argmax(prediction)
#         return CATEGORIES[class_idx]
#     else:
#         return "Not enough frames"

def predict_video(video_path):
    frames_with_flow = preprocess_video(video_path)
    if frames_with_flow:
        frames, _ = zip(*frames_with_flow)  # We ignore the flow component

        yolo_features = extract_features_with_yolo(frames)

        yolo_features = np.repeat(yolo_features[:, np.newaxis, np.newaxis, :], IMG_SIZE, axis=1)
        yolo_features = np.repeat(yolo_features, IMG_SIZE, axis=2)

        frames = np.stack(frames, axis=0)
        combined_features = np.concatenate([frames, yolo_features], axis=-1)

        combined_features = np.expand_dims(combined_features, axis=0)

        prediction = model.predict(combined_features)[0]  # shape: (num_classes,)
        class_idx = np.argmax(prediction) 
        confidence = float(prediction[class_idx])*100  # Confidence score for the predicted class

        return CATEGORIES[class_idx], confidence
    else:
        return "Not enough frames", 0.0


# --- Example Usage ---
# video_file = '202503211710024.mp4'  # Replace with your test video path
# result = predict_video(video_file)
# print(f'The video is classified as: {result}')


# video_file = '202503200003375.mp4'  # Replace with your test video path
# result, confidence = predict_video(video_file)
# print(f'The video is classified as: {result} (Confidence: {confidence:.2f})')



def main():
    video_file = ''  # Replace with your video path
    result, confidence = predict_video(video_file)
    print(f'The video is classified as: {result} (Confidence: {confidence:.2f})')

if __name__ == "__main__":
    main()