# 🫁 Lung Nodule Detection and Localization using JSRT Dataset

## 🌟 Project Overview

This project implements machine learning and deep learning techniques to perform two crucial tasks in medical image analysis: **Classification** (identifying if a patient has a lung nodule) and **Localization** (predicting the precise coordinates of the nodule).

The goal is to develop and compare models using different feature extraction and modeling approaches, from traditional KNN to advanced Neural Networks.

**⚠️ IMPORTANT DISCLAIMER:**
This project is for educational and portfolio purposes only. It uses publicly available data and models and is **not intended** for real-world clinical use or medical diagnosis. Accuracy claims are based solely on the JSRT dataset metrics.

## ⚙️ Methodology & Structure

The codebase is structured to enforce a clean separation of concerns:

1.  **`pre_processing.py`**: Handles all data loading and image manipulation.
2.  **`knn_classification.py`**: Implements K-Nearest Neighbors for the binary classification task.
3.  **`knn_regression.py`** (To be added): Implements K-Nearest Neighbors for the nodule coordinate localization task.
4.  **`deep_learning_classification.py`** (To be added): Implements a Neural Network (likely a Convolutional Neural Network - CNN) for classification.

### Data Source

* **Dataset:** JSRT (Japanese Society of Radiological Technology) database.
* **Input Data:** 254 chest X-ray images (154 nodule cases, 93 non-nodule cases) provided as binary `.IMG` files.
* **Dimensions:** $2048 \times 2048$ pixels, 12-bit grayscale depth.
* **Labels:** Binary presence (0 or 1) for classification; $(x, y)$ coordinates for localization.

### Pre-processing Techniques Implemented

The `pre_processing.py` module includes several sophisticated image filters and techniques necessary to clean and standardize the high-resolution medical images:

* **Color/Scaling:** MinMax Scaling to standard 0-255 range.
* **Size Transformation:** Resizing capabilities (e.g., down to $256 \times 256$) for faster computation/Deep Learning.
* **Histogram Methods:** Standard Histogram Equalization (HED) and Contrast Limited Adaptive Histogram Equalization (CLAHE).
* **Filtering:** Advanced edge detection and feature extraction using **Sobel**, **Meijering**, **Niblack/Sauvola** thresholds, and **Median** filters.
* **Morphology:** Dilation-based reconstruction to highlight features by removing the background.

## 💾 Setup and Installation

### 1. Clone the Repository

```bash
git clone <https://github.com/alizargarzadeh/Lung-Cancer-Detection/>
cd lung-cancer-detection
