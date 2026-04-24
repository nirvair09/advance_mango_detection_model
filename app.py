import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "exported_models"
DETECTION_MODEL_PATH = EXPORT_DIR / "best.pt"
CLASSIFICATION_MODEL_PATH = EXPORT_DIR / "best_classifier.keras"
METADATA_PATH = EXPORT_DIR / "model_metadata.json"

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
CLASS_NAMES = metadata.get("class_names", ["raw", "ripe", "overripe"])
DETECTION_CONF_THRESH = metadata.get("detection_conf_threshold", 0.5)
CLASSIFICATION_CONF_THRESH = metadata.get("classification_conf_threshold", 0.6)
CROP_MARGIN = metadata.get("crop_margin", 10)
IMAGE_SIZE = tuple(metadata.get("classifier_image_size", [224, 224]))
BEST_CLASSIFIER_NAME = metadata.get("best_classifier", "unknown")


@st.cache_resource
def load_models():
    detection_model = YOLO(str(DETECTION_MODEL_PATH))
    classification_model = load_model(CLASSIFICATION_MODEL_PATH)
    return detection_model, classification_model


def detect_mangoes(model, img, conf_thresh=DETECTION_CONF_THRESH):
    results = model.predict(img, conf=conf_thresh, verbose=False)
    boxes = []

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, conf in zip(xyxy, confidences):
            if float(conf) < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            boxes.append((x1, y1, x2, y2, float(conf)))

    return boxes


def crop_mango(img, box, margin=CROP_MARGIN):
    x1, y1, x2, y2, _ = box
    height, width = img.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)
    return img[y1:y2, x1:x2]


def preprocess_crop(crop):
    resized = cv2.resize(crop, IMAGE_SIZE).astype("float32") / 255.0
    return np.expand_dims(resized, axis=0)


def classify_mango(model, crop, conf_thresh=CLASSIFICATION_CONF_THRESH):
    prediction = model.predict(preprocess_crop(crop), verbose=False)[0]
    confidence = float(np.max(prediction))
    label_idx = int(np.argmax(prediction))

    if confidence < conf_thresh:
        return "uncertain", confidence

    return CLASS_NAMES[label_idx], confidence


def process_image(img, det_model, clf_model, det_thresh, cls_thresh):
    annotated = img.copy()
    boxes = detect_mangoes(det_model, annotated, conf_thresh=det_thresh)

    if not boxes:
        return None, {
            "message": "No mango detected",
            "counts": {name: 0 for name in CLASS_NAMES},
            "results": [],
            "avg_detection_conf": 0.0,
            "avg_combined_conf": 0.0,
        }

    counts = {name: 0 for name in CLASS_NAMES}
    results_table = []
    combined_scores = []

    for index, box in enumerate(boxes, start=1):
        crop = crop_mango(annotated, box)
        if crop.size == 0:
            continue

        label, cls_conf = classify_mango(clf_model, crop, conf_thresh=cls_thresh)
        x1, y1, x2, y2, det_conf = box
        combined_conf = float(det_conf * cls_conf)
        combined_scores.append(combined_conf)

        if label != "uncertain":
            counts[label] += 1

        results_table.append(
            {
                "id": index,
                "label": label,
                "detection_conf": round(float(det_conf), 4),
                "classification_conf": round(float(cls_conf), 4),
                "combined_conf": round(combined_conf, 4),
            }
        )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} | det:{det_conf:.2f} cls:{cls_conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    return annotated, {
        "message": "ok",
        "counts": counts,
        "results": results_table,
        "avg_detection_conf": float(np.mean([box[4] for box in boxes])) if boxes else 0.0,
        "avg_combined_conf": float(np.mean(combined_scores)) if combined_scores else 0.0,
    }


def main():
    st.set_page_config(page_title="Mango Detection and Ripeness App", layout="centered")
    st.title("Mango Detection and Ripeness App")
    st.caption(f"Best classifier selected during training: {BEST_CLASSIFIER_NAME}")
    st.write("Upload an image to detect mangoes and classify ripeness for each detected fruit.")

    if not DETECTION_MODEL_PATH.exists() or not CLASSIFICATION_MODEL_PATH.exists():
        st.warning("Exported models were not found. Run the notebook training and export cells first.")
        st.stop()

    detection_model, classification_model = load_models()

    det_thresh = st.slider("Detection confidence threshold", 0.1, 0.95, float(DETECTION_CONF_THRESH), 0.05)
    cls_thresh = st.slider("Classification confidence threshold", 0.1, 0.95, float(CLASSIFICATION_CONF_THRESH), 0.05)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image_array, caption="Uploaded image", channels="BGR", use_container_width=True)

    processed_image, summary = process_image(image_array, detection_model, classification_model, det_thresh, cls_thresh)

    if processed_image is None:
        st.error(summary["message"])
        return

    st.image(processed_image, caption="Detection and ripeness result", channels="BGR", use_container_width=True)
    st.write("Ripeness counts:", summary["counts"])
    st.write(f"Average detection confidence: {summary['avg_detection_conf']:.2f}")
    st.write(f"Average combined confidence: {summary['avg_combined_conf']:.2f}")
    st.dataframe(pd.DataFrame(summary["results"]), use_container_width=True)


if __name__ == "__main__":
    main()
