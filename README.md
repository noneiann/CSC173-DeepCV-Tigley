# Real-Time Philippine Coin Detection & Counting System

**CSC173 Intelligent Systems Final Project**  
_Mindanao State University - Iligan Institute of Technology_  
**Student:** Rey Iann V. Tigley, 2022-0224  
**Semester:** AY 2025-2026 Sem 1  
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org) [![YOLOv12](https://img.shields.io/badge/YOLO-v12s-green)](https://github.com/ultralytics/ultralytics)

## Abstract

Manual coin counting is time-consuming and error-prone, particularly in retail and banking environments in the Philippines. This project develops a real-time Philippine coin detection and counting system using YOLOv12s, targeting 1, 5, 10, and 20 peso denominations. Due to limited availability of annotated coin datasets, a synthetic data generation pipeline was created that composites coin images onto diverse texture backgrounds from the Describable Textures Dataset (DTD). Extensive augmentations including low-light simulation, specular highlights, and geometric transforms improve model robustness. The trained model achieves real-time detection via webcam with automatic value calculation. Key challenges include visual similarity between new 5 and 10 peso bimetallic coins and performance degradation in poor lighting conditions. This work demonstrates the viability of synthetic data pipelines for object detection when real-world annotated data is scarce.

## Table of Contents

- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction

### Problem Statement

Coin counting and sorting is a common task in Philippine retail stores, banks, and transportation systems. Manual counting is slow and prone to human error, especially with high coin volumes. Existing coin counting machines are expensive and not widely accessible. A computer vision-based solution using commodity webcams could provide an affordable alternative for small businesses and personal use in Mindanao and beyond.

### Objectives

- Achieve reliable detection of Philippine peso coins (1, 5, 10, 20 pesos) in real-time
- Develop a synthetic dataset generation pipeline to overcome limited training data
- Integrate detection with automatic value counting logic
- Deploy as a desktop application with adjustable settings for varying conditions

## Related Work

Recent advancements in computer vision have been largely driven by single-stage object detection architectures, most notably the YOLO (You Only Look Once) family, which optimizes trade-offs between inference speed and accuracy for real-time applications [1]. In the specific domain of numismatics, prior research has explored various methodologies for coin recognition. For instance, Roomi and Rajee utilized neural networks to classify coins based on extracted features [4], while Conn and Arandjelovic focused on the more complex task of recognizing ancient coins "in the wild" using robust preprocessing and normalization techniques [5]. However, these existing studies often rely on traditional Convolutional Neural Networks (CNNs) or template matching and predominantly utilize datasets of United States, European, or historical currencies, leaving a significant gap in research regarding modern Philippine denominations.

The development of a robust detection system for Philippine currency is compounded by the lack of publicly available, annotated datasets. To overcome the "cold start" problem of data scarcity, this project leverages synthetic data generation, a technique validated by Tremblay et al., who demonstrated that domain randomization compositing objects against diverse backgrounds, can effectively bridge the "reality gap" for deep learning models [2]. By using the Describable Textures Dataset (DTD) [3] to provide high-variance backgrounds, this study aims to generate a training set that forces the model to learn robust feature representations of the coins themselves, rather than overfitting to environmental context, thereby offering a novel approach to counting Philippine coins in unconstrained environments.

## Methodology

### Dataset

- **Source**: 33 coin images (heads/tails, old/new currency) + DTD texture backgrounds
- **Synthetic Generation**: 10,000 composite images with randomized coin placement
- **Split**: 80/10/10 train/val/test
- **Preprocessing**: Augmentation pipeline applied both during generation and training, images resized to 640x640.

| Coin Class | Diameter (mm) | Images |
| ---------- | ------------- | ------ |
| 1 Peso     | 24.0          | 4      |
| 5 Pesos    | 25.0          | 6      |
| 10 Pesos   | 26.5          | 4      |
| 20 Pesos   | 27.0          | 2      |

### Architecture

- **Backbone**: YOLOv12s (small variant for faster inference)
- **Head**: YOLO detection layers with 4-class output
- **Input**: 640×640 RGB images

### Augmentations

| Category  | Techniques                                                               |
| --------- | ------------------------------------------------------------------------ |
| Color     | HSV jitter, brightness/contrast, low-light simulation, color temperature |
| Lighting  | Specular highlights, gradient shadows, spotlight effects                 |
| Noise     | Gaussian noise, Gaussian blur                                            |
| Geometric | Rotation (360°), flip, shear, perspective transform                      |

### Training Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Batch Size    | 16    |
| Learning Rate | 0.01  |
| Epochs        | 100   |
| Image Size    | 640   |
| Optimizer     | Adam  |
| Mosaic        | 1.0   |
| MixUp         | 0.15  |

## Experiments & Results

### Metrics

| Model                     | mAP@0.5   | Precision | Recall    | mAP@0.5-95 | Inference (ms) |
| ------------------------- | --------- | --------- | --------- | ---------- | -------------- |
| **YOLOV12s (Fine Tuned)** | **0.990** | **0.975** | **0.962** | **0.874**  | **~15**        |

_Note: Metrics to be updated after final training run_

### Per-Class Performance

| Class    | Detection Rate | Common Errors                         |
| -------- | -------------- | ------------------------------------- |
| 1 Peso   | Good           | Occasional size confusion             |
| 5 Pesos  | Moderate       | Confused with 10 Pesos (new currency) |
| 10 Pesos | Moderate       | Confused with 5 Pesos (new currency)  |
| 20 Pesos | Good           | Less common, distinct appearance      |

### Visualizations

#### Training loss over Epochs

![alt text](<models/final model/results.png>)

#### F1 Confidence

![alt text](<models/final model/BoxF1_curve.png>)

#### Precision Confidence

![alt text](<models/final model/BoxP_curve.png>)

#### Precision Recall

![alt text](<models/final model/BoxPR_curve.png>)

#### Confusion Matrix

![alt text](<models/final model/confusion_matrix.png>)

### Demo

Video: [CSC173_Tigley_Final.mp4](https://drive.google.com/file/d/1xVH1vi9R2z7nOuf_7yaSHexEqeUBM90j/view?usp=sharing)

## Discussion

### Strengths

- Real-time detection suitable for interactive applications
- Synthetic data pipeline enables training without manual annotation
- Adjustable brightness/contrast helps adapt to lighting conditions
- Balanced sampling ensures equal representation of all coin classes

### Limitations

- **5 vs 10 Peso Confusion**: New bimetallic 5 and 10 peso coins are visually similar
- **Low-Light Performance**: Detection accuracy degrades significantly in poor lighting
- **Limited Source Diversity**: Only ~30 source coin images limits real-world generalization
- **Synthetic-to-Real Gap**: Model trained on synthetic composites may underperform on real webcam images

### Insights

- Low-light augmentation significantly improved robustness (+improvement TBD)
- Coin-specific augmentations (tint, wear, brightness) increase effective dataset diversity
- Mosaic augmentation during YOLO training provides additional scene variety

## Ethical Considerations

- **Bias**: Dataset focused on peso coins; does not generalize to other currencies
- **Privacy**: No personal data collected; webcam feed processed locally
- **Economic Impact**: Potential to reduce employment in manual counting roles
- **Misuse**: System could theoretically be adapted for counterfeit detection, requiring responsible deployment

## Conclusion

This project demonstrates a practical approach to Philippine coin detection using YOLOv12 trained on synthetically generated data. The system achieves real-time detection with automatic value calculation, addressing the need for affordable coin counting solutions.

### Future Directions

1. **Capture Real Training Data**: Photograph coins under diverse real-world conditions to close synthetic-to-real gap
2. **Expand Coin Classes**: Add centavo coins (1¢, 5¢, 10¢, 25¢) for complete Philippine currency coverage
3. **Edge Deployment**: Optimize for Raspberry Pi or mobile deployment
4. **Active Learning**: Use model predictions to bootstrap real-world annotation

## Installation

1. Clone repo: `git clone https://github.com/yourusername/CSC173-DeepCV-Tigley.git`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: Place trained model in `models/best.pt`
4. Run app: `python webcam_app.py`

**requirements.txt:**

```
torch>=2.0
ultralytics
opencv-python
numpy
pyyaml
```

## References

[1] Jocher, G., et al. "YOLOv8," Ultralytics, 2023. https://github.com/ultralytics/ultralytics  
[2] Tremblay, J., et al. "Training Deep Networks with Synthetic Data: Bridging the Reality Gap," CVPR Workshops, 2018.  
[3] Cimpoi, M., et al. "Describing Textures in the Wild," CVPR, 2014. https://www.robots.ox.ac.uk/~vgg/data/dtd/  
[4] S. M. M. Roomi and R. B. J. Rajee, “Coin detection and recognition using neural networks,” 2015 International Conference on Circuits, Power and Computing Technologies [ICCPCT-2015], vol. 4, pp. 1–6, Mar. 2015, doi: 10.1109/iccpct.2015.7159434.
[5] B. Conn and O. Arandjelovic, “Towards computer vision based ancient coin recognition in the wild — automatic reliable image preprocessing and normalization,” 2017 International Joint Conference on Neural Networks (IJCNN), pp. 1457–1464, May 2017, doi: 10.1109/ijcnn.2017.7966024.

## GitHub Pages

View this project site: [https://yourusername.github.io/CSC173-DeepCV-Tigley/](https://yourusername.github.io/CSC173-DeepCV-Tigley/)
