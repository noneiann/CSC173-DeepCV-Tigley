# CSC173 Deep Computer Vision Project Proposal

**Student:** Rey Iann V. Tigley, 2022-0224
**Date:** 11/12/2025

## 1. Project Title

Automated Optical Coin Counting via Synthetic Data Generation

## 2. Problem Statement

Micro-small and Medium enterprises (MSMEs) in the Philippines rely heavily on cash transactions. However, manual reconciliation of daily earnings, specifically loose change, is tedious, time-consuming, and prone to human error. While hardware-based coin sorters exist, they are often bulky, expensive, and inaccessible to the average sari-sari store owner or jeepney driver.

## 3. Objectives

- To develop a Synthetic Data Generation pipeline that automates the creation and annotation of thousands of training images for Philippine coins (1, 5, 10, 20 PHP) using simple 2D compositing.
- To train a YOLOv12 object detection model solely on synthetic data and evaluate its "Sim-to-Real" transfer performance on real-world photos.
- To create a prototype system that outputs the total monetary value of a pile of coins with at least 90% accuracy in under 2 seconds.

## 4. Dataset Plan

- Sources:
  - https://universe.roboflow.com/philippine-peso-coin-counter-xbjpx/philippine-peso-coin-counter/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true, 53
  - Synthetic Dataset: Generated via Python script using "Table-Top" textures (wood, plastic, concrete) from the Describable Textures Dataset (DTD). Background from https://www.kaggle.com/datasets/jmexpert/describable-textures-dataset-dtd/data. Target: 1,500 images.
- Classes: ['1_CENTAVO' , '5_CENTAVOS', '10_CENTAVOS', '25_CENTAVOS', '1_PESO', '5_PESOS', '10_PESOS', '20_PESOS']
- Acquisition:

  - Integration: The two datasets will be merged. Main dataset be the synthetic, with added real life datasets for better precision

  - Domain Transformation: A python cv script for compositing the coins to the backgrounds.

  - Balancing: Data augmentation will be applied especially shearing and perspective changing.

## 5. Technical Approach

- Architecture Pipeline:
  - Input: RGB Video Frame ($640 \times 640$).
  - Data Preparation: Background + Random coin placement.
  - Inference: YOLOv12s. This architecture is selected for its lightweight parameter count (9.3M), enabling high frame rates on edge devices.
  - Post-Processing logic: Software that maps each class to its respective value to count the total amount of money.
- Model: YOLOv12s
- Framework: PyTorch (via Ultralytics YOLOv12), OpenCV.
- Hardware: Google Colab with A100 GPU

## 6. Expected Challenges & Mitigations

- Visual Similarity (Inter-Class Ambiguity). The NGC 1-Peso and 5-Peso coins are both silver and visually similar in diameter, leading to confusion.

  Mitigation: Strict Scale Constraints. The synthetic generator will enforce consistent relative scaling so the model learns size relationships, not just texture.

- High Occlusion (The "Pile" Problem). In a real scenario, coins overlap significantly.

  Mitigation: "NMS Tuning." Adjust the Non-Maximum Suppression (NMS) threshold to be more permissive, allowing overlapping bounding boxes to exist so the model doesn't delete coins that are "hiding" behind others.

- Specular Reflection (Glare). Coins are metallic and reflective; flash photography can wash out features (make them look white).

  Mitigation: Data Augmentation. Apply random "Brightness/Contrast" jitter and "White Noise" during the synthetic generation process to teach the model to recognize coins even when they are partially washed out by glare.
