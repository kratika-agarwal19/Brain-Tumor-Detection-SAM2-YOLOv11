# Brain-Tumor-Detection-SAM2-YOLOv11# 🧠 NeuroScan AI: Triple-Model Brain Tumor Suite

**NeuroScan AI** is a professional diagnostic tool that integrates **VGG16**, **YOLOv11**, and **SAM2** to provide end-to-end analysis of brain MRI scans.

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

### **2. Precision & Recall Analysis**
![Precision](Precision.jpeg) 
![Recall](recall.png)
![F1 Score](f1%20score.jpg)

---

## 🖼️ Diagnostic Results (Live Dashboard)
![Website](website%20image.jpg)

### **Pathological Segmentation & Severity Analysis**
![Segmentation](segmentation%20result%20%20with%20severity%20and%20area%20ratio.jpg)

---

## 📂 Model Files & Weights
The system uses the following pre-trained weights (ensure these are in your main directory):
* `final_weights.weights.h5` - **VGG16 Engine** (Pathology Classification)
* `best%20(5).pt` - **YOLOv11 Engine** (Tumor Localization)
* `sam2_b.pt` - **SAM2 Engine** (Precision Segmentation)

---

## ⚙️ Installation & Usage
1. **Clone the Project:**
   ```bash
   git clone [https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git](https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git)
   cd Brain-Tumor-Detection-SAM2-YOLOv11
Install Dependencies:
pip install -r requirements.txt
Run the Clinical Dashboard:
streamlit run app.py
