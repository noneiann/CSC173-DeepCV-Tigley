# Privacy-Preserving Crowd Behavior & Attribute Analysis via Thermal-Normalized Representation

**CSC173 Intelligent Systems Final Project** _Mindanao State University - Iligan Institute of Technology_ **Student:** Rey Iann V. Tigley, 2022-0224  
**Semester:** AY 2025-2026 Sem 1  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org) [![YOLOv12](https://img.shields.io/badge/YOLO-v12s-green)](https://github.com/ultralytics/ultralytics)

## Abstract

Standard surveillance systems often rely on facial recognition or invasive detail to analyze crowd dynamics, raising significant privacy concerns. This project proposes a **Privacy-Preserving Feature Extraction System** that transforms standard RGB video into a **Thermal-Normalized Representation**, stripping identity while preserving geometric and behavioral cues. Utilizing a **YOLOv12s** model, the system detects and categorizes individuals based on observable physical features—specifically **Posture (Standing, Crouching, Lying)**, **Relative Size (Height/Build)**, and **Spatial Density**—without assigning demographic labels. Key contributions include a domain-agnostic input pipeline and a focus on "Soft Biometrics" for emergency situational awareness (e.g., detecting fallen individuals or overcrowding) without PII storage.

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

In emergency scenarios like smoke-filled rooms, blackounts, in public monitoring, knowing _who_ someone is matters less than knowing _their state_. Is someone lying down (injured)? Is the crowd crushing together? Standard RGB cameras capture unnecessary identity details (faces, clothing brands) that violate privacy. Existing privacy tools (blurring) often destroy the geometric edges needed to determine these physical states.

### Objectives

- **Privacy Enforcement:** Anonymize visual input via "Pseudo-Thermal" transformation (RGB → Thermal-Normalized) to remove race, gender, and facial identity.
- **Feature Extraction:** Accurately classify physical states: **Standing vs. Crouching/Lying Down** and **Relative Size** (Small vs. Large entities) for spatial analysis.
- **Geometric Robustness:** Utilize horizon-line heuristics to distinguish between "Small Person" (feature) and "Distant Person" (perspective).
- **Real-Time Insight:** Deploy a lightweight YOLOv12s model to map crowd density and behavioral anomalies at >30 FPS.

## Related Work

## Methodology

### Dataset

- **Sources:** Aggregated from Human Detection Dataset, INRIA Person, and behavior-specific subsets (fallen person datasets).
- **Size:** ~2,331 images total.
- **Preprocessing:** - **Domain Transformation:** RGB → Grayscale → CLAHE → `cv2.applyColorMap(COLORMAP_JET)`.
  - **Annotation Update:** Remapped "Adult/Child" labels to feature-based classes.
- **Target Features (Classes):**
  - **Primary States:** `Person-Standing`, `Person-Crouching`, `Person-LyingDown`.
  - **Secondary Attributes (Inferred):** `Size-Small`, `Size-Large` (based on bounding box geometry vs. horizon).
