# Brain-Tumor-Detection-SAM2-YOLOv11
NeuroScan AI: A hybrid DL pipeline for Brain Tumor Analytics. Uses VGG16 for classification, YOLOv11x for real-time localization, and SAM2 for zero-shot instance segmentation. Features an automated severity scaling logic (Area Ratio) to provide quantitative clinical insights. Optimized for high precision (1.0) and recall (0.96) in neuro-oncology.
# 🧠 NeuroScan AI: Triple-Model Brain Tumor Suite

**NeuroScan AI** is a professional diagnostic tool that integrates **VGG16**, **YOLOv11**, and **SAM2** to provide end-to-end analysis of brain MRI scans, from classification to volumetric severity scaling.

---

## 🚀 Key Features
* **Triple-Engine Pipeline:** VGG16 (Classification) + YOLOv11 (Detection) + SAM2 (Segmentation).
* **Automated Severity Scaling:** Calculates **Tumor-to-Brain Areal Ratio** (Safe, Low, Moderate, Critical).
* **Real-time Inference:** Results in ~0.4s using optimized deep learning weights.
* **Clinical Dashboard:** Streamlit-based UI for easy medical data interaction.

---

## 📊 Model Performance (50 Epochs)

### **1. Training Convergence & Losses**
![Losses](losses.jpeg)
*The downward trend in Box and Class loss confirms successful learning over 50 epochs.*

### **2. Precision & Recall Analysis**
![Precision](Precision.jpeg) 
![Recall](recall.png)
*Achieved **0.98 Precision**, ensuring high reliability and minimal false positives in diagnosis.*

---

## 🖼️ Diagnostic Results

### **1. Clinical Dashboard Interface**
![Website](website%20image.jpg)

### **2. Pathological Segmentation & Severity**
![Segmentation](segmentation%20result%20%20with%20severity%20and%20area%20ratio.jpg)
*SAM2 tracing exact tumor boundaries with automated risk staging.*

---

## 📂 Project Structure
* `app.py`: Main Streamlit Dashboard.
* `final_weights.weights.h5`: VGG16 Classification model.
* `best (5).pt`: YOLOv11 Detection model.
* `sam2_b.pt`: SAM2 Segmentation model.
* `requirements.txt`: Project dependencies.

---

## ⚙️ Installation & Usage
1. **Clone:** `git clone https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git`
2. **Install Libs:** `pip install -r requirements.txt`
3. **Run:** `streamlit run app.py`

---

## 📜 License & Disclaimer
Licensed under **MIT License**.
**⚠️ Disclaimer:** This tool is for research purposes and based on the AIIMS Delhi Research Framework. All findings must be reviewed by a certified Neuro-Radiologist.
