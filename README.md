# HEAL ITALIA PROJECT- SPOKE 2
![Pipeline](Logo-Header.png)

# Lung Nodule Segmentation (3D) With SWINUNETR

Volumetric lung nodule segmentation using 3D deep learning models on CT scans.

# Lung Nodule Segmentation (3D)

3D lung nodule segmentation using volumetric CT patches and deep learning models.

## Overview
This repository contains code developed as part of a thesis project on lung nodule segmentation from CT scans using 3D deep learning methods.

## Current Contents
- 3D patch-based training pipeline
- SwinUNETR model
- MONAI-based augmentation
- Hybrid Dice + BCE loss
- Training and validation metrics

## Method
The current implementation uses 3D CT patches stored as `.npy` files and trains a SwinUNETR model for volumetric segmentation.

## Structure
```text
lung-nodule-segmentation-3d/
│
├── README.md
├── .gitignore
├── requirements.txt
├── src/
├── notebooks/
└── results/
```

## Status

Initial version. Further improvements and experiments will be added.
