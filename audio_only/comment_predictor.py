# import json
# import joblib
# from collections import Counter

# # Configuration
# MODEL_PATH = "youtube_comment_classifier.joblib"
# CATEGORIES = ['1', '2', '3', '4', '5', '6']

# # Load model once when module is imported
# model = joblib.load(MODEL_PATH)

# def predict_comments(comment_json_path):
#     """
#     Predict dominant category for a video based on its comments.

#     Args:
#         comment_json_path (str): Path to JSON file containing comments.

#     Returns:
#         Tuple[str, float]: Predicted category label and confidence percentage.
#     """
#     with open(comment_json_path, 'r', encoding='utf-8') as f:
#         comments = json.load(f)

#     predictions = model.predict(comments)
#     predictions = [int(p) for p in predictions]

#     category_counts = Counter(predictions)
#     total = len(comments)

#     if total == 0:
#         return "No comments", 0.0

#     dominant_category, count = category_counts.most_common(1)[0]
#     confidence = (count / total) * 100

#     return CATEGORIES[dominant_category], confidence



# import json
# import joblib
# from collections import Counter

# MODEL_PATH = "youtube_comment_classifier.joblib"
# CATEGORIES = ['1', '2', '3', '4', '5', '6']

# # Load model once
# model = joblib.load(MODEL_PATH)

# def predict_comments(comment_json_path):
#     try:
#         with open(comment_json_path, 'r', encoding='utf-8') as f:
#             comments = json.load(f)

#         if not isinstance(comments, list) or not comments:
#             return "No comments", 0.0

#         # Optional: Remove empty strings or nulls
#         comments = [c for c in comments if isinstance(c, str) and c.strip()]
#         if not comments:
#             return "No valid comments", 0.0

#         predictions = model.predict(comments)
#         predictions = [int(p) for p in predictions]

#         if not predictions:
#             return "No predictions", 0.0

#         category_counts = Counter(predictions)
#         dominant_category, count = category_counts.most_common(1)[0]
#         confidence = (count / len(predictions)) * 100

#         return CATEGORIES[dominant_category], confidence

#     except Exception as e:
#         print(f"[Comment Prediction Error] {e}")
#         return "Error", 0.0




import json
import joblib
from collections import Counter

MODEL_PATH = "youtube_comment_classifier.joblib"

# Load model once
model = joblib.load(MODEL_PATH)

def predict_comments(comment_json_path):
    try:
        with open(comment_json_path, 'r', encoding='utf-8') as f:
            comments = json.load(f)

        if not isinstance(comments, list) or not comments:
            return "No comments", 0.0

        # Optional: Remove empty strings or nulls
        comments = [c for c in comments if isinstance(c, str) and c.strip()]
        if not comments:
            return "No valid comments", 0.0

        predictions = [int(p) for p in model.predict(comments)]

        if not predictions:
            return "No predictions", 0.0

        category_counts = Counter(predictions)
        dominant_category, count = category_counts.most_common(1)[0]
        confidence = (count / len(predictions)) * 100

        # Just return the actual predicted number (as string if needed)
        return str(dominant_category), confidence

    except Exception as e:
        print(f"[Comment Prediction Error] {e}")
        return "Error", 0.0
