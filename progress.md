# CSC173 Deep Computer Vision Project Progress Report

**Student:** [Your Name], [ID]  
**Date:** [Progress Submission Date]  
**Repository:** [https://github.com/yourusername/CSC173-DeepCV-YourLastName](https://github.com/yourusername/CSC173-DeepCV-YourLastName)

## 📊 Current Status

| Milestone           | Status         | Notes                              |
| ------------------- | -------------- | ---------------------------------- |
| Dataset Preparation | ✅ Completed   | [X] images downloaded/preprocessed |
| Initial Training    | ✅ Completed   | [X] epochs completed               |
| Baseline Evaluation | ✅ Completed   | Training ongoing                   |
| Model Fine-tuning   | ✅ in-progress | model is so bad                    |

## 1. Dataset Progress

- **Total images:** 2203
- **Train/Val/Test split:** 80%/10%/10%
- **Classes implemented:** [standing, laying, sitting]
- **Preprocessing applied:** Resize(640), normalization, augmentation (thermalization, contrast, hue, shear)

**Sample data preview:**
![Dataset Sample](datasets\processed_thermal\images\train\0.jpg)

## 2. Training Progress

**Training Curves (so far)**
![Loss Curve](images/loss_curve.png)
![mAP Curve](images/map_curve.png)

**Current Metrics:**
| Metric | Train | Val |
|--------|-------|-----|
| Loss | [0.45] | [0.62] |
| mAP@0.5 | [78%] | [72%] |
| Precision | [0.81] | [0.75] |
| Recall | [0.73] | [0.68] |

## 3. Challenges Encountered & Solutions

| Issue              | Status     | Resolution                           |
| ------------------ | ---------- | ------------------------------------ |
| CUDA out of memory | ✅ Fixed   | Reduced batch_size from 32→16        |
| Class imbalance    | ⏳ Ongoing | Added class weights to loss function |
| Slow validation    | ⏳ Planned | Implement early stopping             |

## 4. Next Steps (Before Final Submission)

- [ ] Complete training (50 more epochs)
- [ ] Hyperparameter tuning (learning rate, augmentations)
- [ ] Baseline comparison (vs. original pre-trained model)
- [ ] Record 5-min demo video
- [ ] Write complete README.md with results
