# CSC173 Deep Computer Vision Project Proposal

**Student:** Rey Iann V. Tigley, 2022-0224
**Date:** 11/12/2025

## 1. Project Title

Privacy-Preserving Crowd Monitoring via Pseudo-Thermal Image Transformation

## 2. Problem Statement

Standard video surveillance systems in public spaces raise significant ethical and legal concerns regarding privacy, as they capture personally identifiable information (PII) such as facial features.Furthermore, emergency response teams often struggle to prioritize vulnerable populations like children in low-visibility conditions. Real thermal cameras are prohibitively expensive for widespread deployment. There is a critical need for a low-cost, software-defined solution that can anonymize visual data into "pseudo-thermal" streams while retaining the ability to distinguish between Adults and Children based on body morphology.

## 3. Objectives

- Construct a synthetic "Pseudo-Thermal" dataset containing Adult and Child classes by applying Inverted Channel Enhancement to standard datasets.
- Train a YOLOv8 model to distinguish between Adult and Child based on bounding box aspect ratios and silhouette features, targeting a mean average precision of 75% and a real-time inference speed of ≥ 30 FPS on standard hardware.
- Implement a robust validation pipeline using an 80/10/10 split (Train/Validation/Test) to monitor overfitting and ensure the model generalizes to unseen "thermal-style" data.
- Implement a post-processing "Ratio Check" algorithm that flags detections as "Child" if their vertical height relative to the scene horizon falls below a dynamic threshold.

## 4. Dataset Plan

- Sources:
  - https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset, 921 images
  - https://universe.roboflow.com/pascal-to-yolo-8yygq/inria-person-detection-dataset/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true, 902 images
  - https://www.kaggle.com/datasets/kaushigihanml/kids-and-adults-detection, 508 images
  - Total: ~2,331 images.
- Classes: ['Adult' , 'Child']
- Acquisition:

  - Integration: The three datasets will be merged. Main dataset will be kids and adults detection dataset, the images from the other two with adults, children or both in the frame will be cherry-picked and annotated manually.

  - Domain Transformation: A Python script using OpenCV (cv2.applyColorMap(cv2.COLORMAP_JET)) will batch-process all 2,331 images into "Pseudo-Thermal" heatmaps.

  - Balancing: Data augmentation will be applied specifically to the Child class to address class imbalance.

## 5. Technical Approach

- Architecture Pipeline:
  - Input: RGB Video Frame ($640 \times 640$).
  - Preprocessing: Grayscale $\rightarrow$ CLAHE (Contrast Limited Adaptive Histogram Equalization) $\rightarrow$ Thermal ColorMap (Anonymization Layer).
  - Inference: YOLOv12s. This architecture is selected for its lightweight parameter count (9.3M), enabling high frame rates on edge devices.
  - Post-Processing logic: Geometric filter where $Ratio = Height / Width$. Detections with adult-like ratios but small sizes (false positives) will be filtered based on y-axis position (depth heuristic).
- Model: YOLOv12s
- Framework: PyTorch (via Ultralytics YOLOv12).
- Hardware: Google Colab with A100 GPU

## 6. Expected Challenges & Mitigations

Scale Ambiguity. (A distinct challenge where an adult standing far away looks the same size as a child standing close).

- Mitigation: Implementation of a "Horizon Line constraint." We will define a virtual floor plane; detections that are small but located low in the image (close to camera) will be flagged as valid "Child" detections, while small detections high in the image (far from camera) will be reclassified as distant Adults.

Loss of Visual Features.

- Mitigation: Training will focus on Silhouette Learning. We will use aggressive random erasing and Canny edge injection during training to force the model to learn body shapes rather than clothing textures.
