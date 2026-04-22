# Mango Detection and Ripeness Classification

This project builds an end-to-end computer vision pipeline for mango analysis. It first detects mangoes in an image using YOLOv8, then classifies the ripeness stage of each detected mango using an EfficientNetB0-based TensorFlow model.

The work is implemented primarily in the notebook [advance_mango_detection.ipynb](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/advance_mango_detection.ipynb), with a professional project report also available as [mango_notebook_report.tex](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/mango_notebook_report.tex) and [advance_mango_detection_report.pdf](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/advance_mango_detection_report.pdf).

## Project Overview

The goal of this project is to solve two practical problems:

1. Detect mangoes in real images.
2. Identify the ripeness stage of each mango as `raw`, `ripe`, or `overripe`.

This is useful in agriculture, post-harvest handling, warehouse inspection, retail sorting, and quality-control workflows where fruit maturity matters.

Instead of treating the whole image as a single label, this project uses a two-stage pipeline:

1. Object detection to locate mangoes.
2. Image classification on cropped mango regions to estimate ripeness.

This design is more realistic because a single image can contain multiple mangoes, background clutter, or no mango at all.

## Problem Statement

Manual fruit inspection is slow, subjective, and difficult to scale. In real-world scenarios, we need a system that can:

- find mangoes in images with different backgrounds and lighting,
- separate fruit from surrounding clutter,
- classify maturity level,
- avoid making confident claims when the prediction is weak,
- return a meaningful result even when no mango is present.

This project addresses that need with a modular deep learning pipeline.

## Solution Summary

The project uses:

- `YOLOv8` for mango detection,
- `EfficientNetB0` with transfer learning for ripeness classification,
- `OpenCV` and `NumPy` for preprocessing,
- `TensorFlow/Keras` for the classifier,
- `Roboflow` for the detection dataset,
- `Kaggle` for the ripeness classification dataset,
- `Streamlit` for deployment as a simple web app.

The final system is designed to:

- detect mango bounding boxes,
- crop mango regions with a small margin,
- preprocess each crop,
- classify ripeness,
- mark low-confidence results as `uncertain`,
- report `No mango detected` when appropriate,
- display visual annotations and confidence summaries.

## Repository Contents

- [advance_mango_detection.ipynb](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/advance_mango_detection.ipynb): main notebook containing the full workflow
- [readme.md](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/readme.md): repository documentation
- [mango_notebook_report.tex](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/mango_notebook_report.tex): Overleaf-ready technical report
- [advance_mango_detection_report.pdf](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/advance_mango_detection_report.pdf): compiled project report PDF

## Workflow

The notebook is organized into a sequence of practical steps.

### 1. Environment Setup

The notebook installs the main libraries:

```bash
pip install ultralytics roboflow tensorflow streamlit opencv-python kaggle
```

These libraries support training, inference, dataset download, app deployment, and preprocessing.

### 2. Project Configuration

The notebook defines:

- dataset paths,
- model paths,
- confidence thresholds,
- class labels,
- crop margin,
- image sizes.

Important settings used in the notebook:

- Detection confidence threshold: `0.5`
- Classification confidence threshold: `0.6`
- Crop margin: `10` pixels
- Classifier image size: `224 x 224`
- Ripeness classes: `raw`, `ripe`, `overripe`

### 3. Robust Inference Utilities

Reusable helper functions are built for:

- mango detection,
- crop extraction,
- crop preprocessing,
- ripeness prediction,
- full image processing.

These functions improve code quality and make the notebook easier to convert into an application.

### 4. Mango Detection Dataset Download

The project downloads a YOLOv8-ready dataset from Roboflow using the Python SDK.

Notebook configuration:

- Workspace: `pigrecipe`
- Project: `mango-7l7ug-napwv-razzf`
- Version: `1`

The dataset is downloaded into:

```text
mango-1/
```

It includes:

- `train/`
- `valid/`
- `test/`
- `data.yaml`

### 5. YOLOv8 Detector Training

The project initializes a lightweight YOLOv8 model:

```python
detector = YOLO("yolov8n.pt")
```

Training settings used:

- Epochs: `50`
- Image size: `640`
- Batch size: `16`

The detector is trained to localize mangoes in images.

### 6. Detector Inference on Test Images

After training, the model is used to predict on the test set and save annotated outputs. The saved notebook output shows that test inference ran on `50` images and produced detections such as:

- `1 Fresh Mango`
- `2 Fresh Mangos`
- `6 Fresh Mangos`

This confirms that the object detection stage was executed successfully in the saved notebook.

### 7. Save Results to Google Drive

The notebook includes an optional Colab-specific step to mount Google Drive and copy training results for persistence:

```python
drive.mount('/content/drive')
```

This is useful because Colab runtimes are temporary.

### 8. Reload Best Detector and Review Results

The notebook includes code to:

- reload the best model weights,
- inspect `results.png`,
- preview prediction images.

This step is included in the notebook design, although it was not executed in the saved notebook state.

### 9. Crop Detected Mangoes

The project creates cropped mango images from detector outputs. Each box is slightly expanded using the crop margin so that the classifier receives a more stable fruit region.

This stage is important because it connects detection to classification.

### 10. Ripeness Classification Dataset Download

The notebook downloads the ripeness dataset from Kaggle:

```bash
kaggle datasets download -d srabon00/mango-ripening-stage-classification
```

The archive is extracted to:

```text
/content/ripeness_data
```

This dataset is used to train the classifier on ripeness labels.

### 11. TensorFlow Data Preparation

The classifier uses `ImageDataGenerator` with:

- rescaling by `1/255`,
- validation split `0.2`,
- target size `224 x 224`,
- batch size `32`.

This creates training and validation data loaders directly from class-organized folders.

### 12. EfficientNetB0 Classifier

The project uses transfer learning with `EfficientNetB0`:

- pretrained on ImageNet,
- top classification head removed,
- backbone frozen initially,
- custom dense layers added for 3-class ripeness prediction.

Classifier head:

- `GlobalAveragePooling2D`
- `Dense(128, activation="relu")`
- `Dense(3, activation="softmax")`

Compilation settings:

- Optimizer: `adam`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`

Training setting:

- Epochs: `10`

### 13. End-to-End Pipeline Test

The notebook includes code to test the full pipeline on one sample test image:

1. detect mangoes,
2. crop each mango,
3. classify ripeness,
4. annotate the image,
5. compute summary statistics.

The summary includes:

- class counts,
- average detection confidence,
- average combined confidence.

### 14. Model Export

The notebook includes export logic for:

- the best YOLO detector weights as `best.pt`,
- the classifier as `mango_classifier.h5`.

These files are intended for later deployment in the Streamlit app.

### 15. Streamlit App Generation

The final notebook cell writes `app.py`, which:

- loads both models,
- accepts uploaded images,
- runs the end-to-end pipeline,
- displays annotated results,
- shows counts and confidence summaries.

This turns the notebook into a basic deployable application.

## Architecture

### High-Level Pipeline

```text
Input Image
   -> YOLOv8 Mango Detection
   -> Bounding Boxes
   -> Crop Each Mango
   -> Preprocess Crop
   -> EfficientNetB0 Ripeness Classification
   -> Confidence Filtering
   -> Final Annotated Output + Summary
```

### Why Two Models Are Used

Two separate models are used because the tasks are different:

- the detector answers: `Where is the mango?`
- the classifier answers: `What is its ripeness stage?`

This is more flexible and more practical than forcing one model to do everything from the full scene directly.

## Core Logic

### Detection

The detector returns bounding boxes only when the confidence is at least `0.5`.

### Cropping

Each mango is cropped with a `10`-pixel margin to avoid tight-box failures.

### Classification

The classifier predicts one of:

- `raw`
- `ripe`
- `overripe`

If the highest softmax confidence is below `0.6`, the label becomes:

- `uncertain`

### No-Mango Handling

If no boxes are detected, the system returns:

```text
No mango detected
```

This avoids false reporting on irrelevant images.

### Combined Confidence

The notebook computes:

```text
combined_confidence = detection_confidence * classification_confidence
```

This provides a compact end-to-end confidence indicator.

## What Was Actually Executed in the Saved Notebook

The saved notebook shows direct evidence that these stages were executed:

- package installation,
- Roboflow dataset download,
- detector training,
- detector test inference,
- dataset folder inspection.

The following stages are present in code but were not executed in the saved notebook:

- best-weight reload and visualization,
- crop generation,
- Kaggle ripeness dataset training flow,
- classifier training,
- full pipeline test,
- model export,
- Streamlit app writing.

This is important for honest reporting. The project design is complete in code, but the saved notebook captures partial execution, mainly for the detection stage.

## Strengths of the Project

- Clear two-stage computer vision pipeline
- Practical real-world problem
- Good use of transfer learning
- Confidence-based decision logic
- Handles multiple mangoes in one image
- Includes `uncertain` fallback label
- Includes `No mango detected` handling
- Includes deployment-oriented Streamlit code
- Modular helper functions improve readability and reuse

## Limitations

- The saved notebook does not include final classifier training outputs.
- No final classifier accuracy report is visible in the saved notebook state.
- No confusion matrix or detailed evaluation report is included yet.
- The detection and ripeness datasets come from different sources, which may introduce domain shift.
- The Roboflow API key is hard-coded in the notebook and should be secured.

## Recommended Improvements

- Execute the full classification stage and save metrics.
- Add confusion matrix, classification report, and validation plots.
- Save detector metrics such as precision, recall, and mAP in the repository.
- Add negative images for stronger `No mango detected` behavior.
- Replace the hard-coded API key with environment variables or secrets.
- Export and version trained models inside a dedicated `models/` folder.
- Add screenshots of predicted outputs and the Streamlit app UI.
- Add reproducible requirements in a `requirements.txt` file.

## How to Run the Project

### Option 1: Run the Notebook in Google Colab

This project is clearly designed for Colab. Open the notebook and run the cells in order.

Main requirements:

- Google Colab runtime
- internet access for dataset download
- Roboflow access
- Kaggle API access

### Option 2: Run Locally

Install dependencies:

```bash
pip install ultralytics roboflow tensorflow streamlit opencv-python kaggle
```

Then run the notebook in Jupyter or adapt the final app into a local script after exporting:

```bash
streamlit run app.py
```

Note: `app.py` is written by the final notebook cell, so it may not exist until that cell is executed.

## Expected Outputs

After full execution, the project is intended to generate:

- trained YOLO detector weights,
- cropped mango images,
- trained ripeness classifier model,
- annotated test predictions,
- confidence summaries,
- a deployable Streamlit app.

Expected exported files include:

- `best.pt`
- `mango_classifier.h5`
- prediction images under the YOLO runs directory

## Use Cases

- Fruit sorting systems
- Agricultural monitoring
- Smart warehouse inspection
- Retail fruit quality assessment
- Educational demonstration of deep learning pipelines
- Portfolio project for computer vision and applied AI

## Dataset and Technology Sources

### Datasets

- Roboflow mango detection dataset via project `mango-7l7ug-napwv-razzf`
- Kaggle dataset: `srabon00/mango-ripening-stage-classification`

### Main Frameworks

- Ultralytics YOLOv8
- TensorFlow / Keras
- OpenCV
- Roboflow SDK
- Streamlit

## References

- Ultralytics YOLO Docs: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- Roboflow Docs: [https://docs.roboflow.com/](https://docs.roboflow.com/)
- Kaggle Dataset: [https://www.kaggle.com/datasets/srabon00/mango-ripening-stage-classification](https://www.kaggle.com/datasets/srabon00/mango-ripening-stage-classification)
- Keras EfficientNet Docs: [https://keras.io/api/applications/efficientnet/](https://keras.io/api/applications/efficientnet/)
- TensorFlow Transfer Learning Guide: [https://www.tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- Streamlit Docs: [https://docs.streamlit.io/](https://docs.streamlit.io/)

## Final Note

This project already has a strong foundation: it combines object detection, classification, practical decision rules, and deployment planning in a single workflow. The next big milestone is to fully execute the remaining notebook stages and record the final experimental results so the project is not only well-designed, but also fully validated.
