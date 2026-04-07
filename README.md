# <p align="center">🧠 NeuroScan AI: Triple-Model Brain Tumor Suite</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <img src="https://img.shields.io/badge/Framework-Ultralytics%20%7C%20TensorFlow-orange">
  <img src="https://img.shields.io/badge/Models-VGG16%20%7C%20YOLOv11%20%7C%20SAM2-green">
  <img src="https://img.shields.io/badge/License-MIT-red.svg">
</p>

---

## 🚀 Overview
**NeuroScan AI** is a professional-grade diagnostic assistant that bridges the gap between raw MRI data and clinical insights. It uses a **Hybrid Neural Pipeline** to classify, locate, and segment brain tumors with high-precision scoring.

---

## 📊 Performance Analytics (50 Epochs)

### **Training Metrics & Loss**
<p align="center">
  <img src="losses.jpeg" width="800">
</p>
*The consistent downward trend in Box and Class loss confirms successful model convergence over 50 epochs.*

### **Precision, Recall & F1 Analysis**
<p align="center">
  <img src="Precision.jpeg" width="300"> 
  <img src="recall.png" width="300">
  <img src="f1%20score.jpg" width="300">
</p>

---

## 🖼️ Clinical Dashboard & Results

### **User Interface**
<p align="center">
  <img src="website%20image.jpg" width="900">
</p>

### **Pixel-Level Segmentation & Severity Scaling**
<p align="center">
  <img src="segmentation%20result%20%20with%20severity%20and%20area%20ratio.jpg" width="900">
</p>

---

## 📂 Project Architecture
* `app.py`: Main Streamlit Application.
* `final_weights.weights.h5`: VGG16 weights for Pathology Classification.
* `best%20(5).pt`: YOLOv11 weights for Tumor Localization.
* `sam2_b.pt`: SAM2 weights for Instance Segmentation.
* `requirements.txt`: List of required libraries.

---

## ⚙️ Quick Start Guide

1. **Clone the Project:**
   ```bash
   git clone [https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git](https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git)
   cd Brain-Tumor-Detection-SAM2-YOLOv11
