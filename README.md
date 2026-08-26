# Fracture Detection AI

An AI-powered X-ray fracture detection system with visual explainability, built to compare model performance against radiologist interpretation as part of an ongoing research study.

## Overview
- **Model:** DenseNet-121 (transfer learning), fine-tuned for binary fracture classification
- **Dataset:** [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012) — 4,083 radiologist-annotated musculoskeletal X-rays
- **Explainability:** Grad-CAM heatmaps showing which image regions drove each prediction
- **Backend:** FastAPI serving predictions and Grad-CAM overlays
- **Frontend:** Drag-and-drop web interface for uploading X-rays and viewing results

## Results (held-out test set)
- Accuracy: 83%
- AUC: 0.86
- Decision threshold tuned to prioritize recall (catching more true fractures), given the clinical cost asymmetry between missed fractures and false alarms

## Key findings
- Identified and corrected an overfitting issue via data augmentation and early stopping
- Grad-CAM analysis revealed the model may partially rely on visible surgical hardware as a shortcut signal correlated with fracture labels — documented as a limitation for future work
- Built a custom stratified train/valid/test split after discovering the dataset's official split only included fracture-positive images

## Project structure
