import os
import json
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import base64

app = Flask(__name__)
CORS(app)

# Paths (Relative to the project root)
BASE_DIR = Path(__file__).resolve().parent.parent
EXPORT_DIR = BASE_DIR / "exported_models"
DETECTION_MODEL_PATH = EXPORT_DIR / "best.pt"
CLASSIFICATION_MODEL_PATH = EXPORT_DIR / "best_classifier.keras"
METADATA_PATH = EXPORT_DIR / "model_metadata.json"

# Default settings if metadata not found
CLASS_NAMES = ["raw", "ripe", "overripe"]
DETECTION_CONF_THRESH = 0.5
CLASSIFICATION_CONF_THRESH = 0.6
IMAGE_SIZE = (224, 224)

# Global models
det_model = None
clf_model = None

def load_models():
    global det_model, clf_model, CLASS_NAMES, DETECTION_CONF_THRESH, CLASSIFICATION_CONF_THRESH, IMAGE_SIZE
    
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            CLASS_NAMES = metadata.get("class_names", CLASS_NAMES)
            DETECTION_CONF_THRESH = metadata.get("detection_conf_threshold", DETECTION_CONF_THRESH)
            CLASSIFICATION_CONF_THRESH = metadata.get("classification_conf_threshold", CLASSIFICATION_CONF_THRESH)
            IMAGE_SIZE = tuple(metadata.get("classifier_image_size", IMAGE_SIZE))

    if DETECTION_MODEL_PATH.exists() and CLASSIFICATION_MODEL_PATH.exists():
        det_model = YOLO(str(DETECTION_MODEL_PATH))
        clf_model = load_model(str(CLASSIFICATION_MODEL_PATH))
        print("Models loaded successfully.")
        return True
    else:
        print(f"Models not found at {DETECTION_MODEL_PATH} or {CLASSIFICATION_MODEL_PATH}")
        return False

# Load models at startup
models_ready = load_models()

def process_image_logic(img):
    if not models_ready:
        return {"error": "Models not loaded. Please ensure best.pt and best_classifier.keras are in exported_models/"}

    # Detection
    results = det_model.predict(img, conf=DETECTION_CONF_THRESH, verbose=False)
    boxes = []
    for result in results:
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confidences):
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2, float(conf)))

    if not boxes:
        return {"detected": False, "message": "No mangoes detected", "counts": {name: 0 for name in CLASS_NAMES}}

    counts = {name: 0 for name in CLASS_NAMES}
    annotated = img.copy()

    for x1, y1, x2, y2, conf in boxes:
        # Crop and Classify
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        resized = cv2.resize(crop, IMAGE_SIZE).astype("float32") / 255.0
        prediction = clf_model.predict(np.expand_dims(resized, axis=0), verbose=False)[0]
        cls_conf = float(np.max(prediction))
        label_idx = int(np.argmax(prediction))
        
        label = CLASS_NAMES[label_idx] if cls_conf >= CLASSIFICATION_CONF_THRESH else "uncertain"
        
        if label != "uncertain":
            counts[label] += 1

        # Annotate
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} {cls_conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "detected": True,
        "counts": counts,
        "image": img_base64
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    result = process_image_logic(img)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
