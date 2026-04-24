# Mango Detection and Ripeness Classification

This project is an end-to-end computer vision pipeline for mango analysis. It detects mangoes in images using YOLOv8, then classifies ripeness using transfer learning and compares multiple classifier backbones, including EfficientNetB0 and MobileNetV2.

The repository now supports a more final-year-project-style workflow with:

- detector training and review,
- class distribution analysis,
- explicit train/validation/test splits,
- classifier comparison,
- learning curves and confusion matrices,
- exported deployment artifacts,
- a standalone Streamlit app.

## Repository Contents

- [advance_mango_detection.ipynb](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/advance_mango_detection.ipynb): upgraded notebook with training, evaluation, graphs, model comparison, and export
- [app.py](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/app.py): standalone Streamlit app for inference
- [requirements.txt](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/requirements.txt): Python dependencies
- [readme.md](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/readme.md): project documentation
- [mango_notebook_report.tex](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/mango_notebook_report.tex): Overleaf-ready technical report
- [advance_mango_detection_report.pdf](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/advance_mango_detection_report.pdf): compiled report PDF

## Problem Statement

The project solves two practical problems:

1. detect mangoes inside real-world images,
2. classify each detected mango as `raw`, `ripe`, or `overripe`.

This is useful for:

- fruit sorting,
- post-harvest analysis,
- warehouse inspection,
- retail quality checking,
- academic computer vision demonstrations.

## Project Pipeline

```text
Input Image
  -> YOLOv8 Mango Detection
  -> Bounding Boxes
  -> Mango Cropping
  -> Ripeness Classification
  -> Confidence Filtering
  -> Annotated Output + Summary Table
```

## Models Used

### Detection

- `YOLOv8n`
- trained on a Roboflow mango detection dataset
- stronger training default in upgraded notebook: `75` epochs with patience-based stopping

### Classification

The project now compares two classifier backbones:

- `EfficientNetB0`
- `MobileNetV2`

Both models are trained with:

- ImageNet pretrained weights,
- frozen-backbone transfer learning,
- callback-based checkpointing,
- learning-rate reduction,
- early stopping,
- light fine-tuning.

## Datasets

### Detection Dataset

Downloaded from Roboflow using:

- Workspace: `pigrecipe`
- Project: `mango-7l7ug-napwv-razzf`
- Version: `1`

### Ripeness Dataset

Downloaded from Kaggle:

- `srabon00/mango-ripening-stage-classification`

The upgraded notebook converts the image folders into a dataframe so class balance can be inspected before training.

## Upgraded Features Added

Compared with the earlier notebook, the upgraded version now includes:

- MobileNetV2 alongside EfficientNetB0
- higher YOLO training epochs
- explicit train/validation/test splitting
- class-weight computation for imbalance handling
- training and validation graphs
- confusion matrices
- classification reports
- backbone comparison table and bar chart
- model metadata export
- standalone `app.py`

## Notebook Workflow

### 1. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install ultralytics roboflow tensorflow streamlit opencv-python kaggle seaborn scikit-learn pandas matplotlib
```

### 2. Detection Training

The notebook downloads the YOLOv8-ready dataset from Roboflow and trains:

```python
detector = YOLO("yolov8n.pt")
```

Recommended detector defaults in the upgraded notebook:

- Epochs: `75`
- Image size: `640`
- Batch size: `16`
- Patience: `20`

### 3. Detector Review

The notebook then:

- reloads best YOLO weights,
- validates on held-out data,
- saves prediction outputs,
- displays `results.png`,
- previews sample predictions.

### 4. Ripeness Dataset Analysis

The notebook:

- downloads the Kaggle dataset,
- builds a dataframe of filepaths and labels,
- plots class distribution,
- creates explicit train, validation, and test splits.

### 5. Classifier Training

The notebook trains and compares:

- `EfficientNetB0`
- `MobileNetV2`

Recommended classifier defaults:

- Initial training epochs: `25`
- Fine-tuning epochs: `8`
- Batch size: `32`
- Dropout: `0.35`
- Class weighting enabled

### 6. Evaluation Outputs

The upgraded workflow now produces:

- class distribution plots,
- accuracy and loss curves,
- confusion matrices,
- classification reports,
- backbone comparison plots,
- end-to-end annotated sample outputs.

### 7. Export

The notebook exports:

- `best.pt`
- `best_classifier.keras`
- per-backbone classifier files
- `classifier_comparison.csv`
- `model_metadata.json`

## Standalone App

The repository now includes [app.py](C:/Users/Rup/Desktop/AI-ML-DL/advance_mango/app.py), which:

- loads exported detector and classifier models,
- allows image upload,
- supports threshold tuning,
- displays annotated outputs,
- shows ripeness counts,
- displays a per-detection result table.

Run it with:

```bash
streamlit run app.py
```

Note: exported models must exist first, so run the notebook training and export cells before launching the app.

## Scope of Further Improvement

The project is now much stronger, but these additions would take it even further:

- add a negative-image dataset for more reliable `No mango detected` behavior,
- add Grad-CAM for classifier interpretability,
- try k-fold validation for more rigorous evaluation,
- log experiments with MLflow or Weights & Biases,
- export to ONNX or TensorFlow Lite,
- create a larger real-world test set with cluttered market images,
- add ablation studies for crop margin, thresholds, and augmentation settings.

## References

- Ultralytics YOLO Docs: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- Roboflow Docs: [https://docs.roboflow.com/](https://docs.roboflow.com/)
- Kaggle Dataset: [https://www.kaggle.com/datasets/srabon00/mango-ripening-stage-classification](https://www.kaggle.com/datasets/srabon00/mango-ripening-stage-classification)
- Keras EfficientNet Docs: [https://keras.io/api/applications/efficientnet/](https://keras.io/api/applications/efficientnet/)
- Keras MobileNetV2 Docs: [https://keras.io/api/applications/mobilenet/](https://keras.io/api/applications/mobilenet/)
- TensorFlow Transfer Learning Guide: [https://www.tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- Streamlit Docs: [https://docs.streamlit.io/](https://docs.streamlit.io/)
