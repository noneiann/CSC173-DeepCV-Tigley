# CSC173 Deep Computer Vision Project Progress Report

**Student:** [Your Name], [ID]  
**Date:** [Progress Submission Date]  
**Repository:** [https://github.com/yourusername/CSC173-DeepCV-YourLastName](https://github.com/yourusername/CSC173-DeepCV-YourLastName)

## 📊 Current Status

| Milestone           | Status      | Notes                                                                                                 |
| ------------------- | ----------- | ----------------------------------------------------------------------------------------------------- |
| Dataset Preparation | ✅ Complete | [X] images downloaded/preprocessed                                                                    |
| Initial Training    | ✅ Complete | [X] 100 epochs completed                                                                              |
| Baseline Evaluation | ✅ Complete | Needs fine tuning                                                                                     |
| Model Fine-tuning   | ✅ Complete | Dataset needs more real life images;                                                                  |
| Dataset Additions   | ✅ Complete | Added more images; 53 images and 296 images                                                           |
| Evaluation          | ✅ Complete | test and val have high values now, but model is still not accurate from phone or webcam video quality |

## 1. Dataset Progress

- **Total images:** 10,000
- **Train/Val/Test split:** 80%/10%/10%
- **Classes implemented:** 1,5,10,20 piso
- **Preprocessing applied:** shear, hue, blur

## 2. Training Progress

**Training Curves (so far)**

**Current Metrics:** UNFINISHED
| Metric | Train | Val |
|--------|-------|-----|
| mAP@0.5 | [78%] | [72%] |
| Precision | [0.81] | [0.75] |
| Recall | [0.73] | [0.68] |
| **YOLOV12s (Fine Tuned)** | **0.990** | **0.975** | **0.962** | **0.874** | **~15** |
**Training Curves (after tuning and additions)**
**Current Metrics:** UNFINISHED
| Metric | Val | Split |
|--------|-------|-----|
| mAP@0.5 | [99%] | [99%] |
| Precision | [0.97] | [0.975] |
| Recall | [0.951] | [0.962] |
| mAP@0.5-95 | [0.819] | [0.874] |

## 3. Challenges Encountered & Solutions

| Issue                        | Status   | Resolution                                                                |
| ---------------------------- | -------- | ------------------------------------------------------------------------- |
| Class imbalance              | ✅ Fixed | Reduced classes with too much appearances                                 |
| Visual Similarity (1₱ vs 5₱) | ✅ Fixed | Enforced consistent relative scaling in synthetic generator               |
| High Occlusion (Coin Piles)  | ✅ Fixed | Tuned NMS threshold to be more permissive for overlapping bounding boxes  |
| Specular Reflection (Glare)  | ✅ Fixed | Applied random brightness/contrast jitter and white noise in augmentation |

## 4. Next Steps (Before Final Submission)

- [x] Record 5-min demo video
- [x] Write complete README.md with results
