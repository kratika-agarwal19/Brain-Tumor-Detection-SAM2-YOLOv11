# 🧠 NeuroScan AI: Advanced Neuro-Oncology Suite
**An Integrated Pipeline for Brain Tumor Classification & Precision Segmentation**

---

## 📊 Performance Analytics & Validation
The system is built upon the **AIIMS Delhi Research Framework**, trained over 50 epochs to ensure high diagnostic reliability.

### 📈 Training Dynamics (Loss & Accuracy)
The following visualization illustrates the convergence of our hybrid models during the training phase:
<p align="center">
  <img src="losses.jpeg" width="850" alt="Training Metrics">
</p>

### 🔬 Statistical Evaluation
Our models are validated using standard clinical metrics including Precision, Recall, and F1-Score to ensure minimal false negatives in tumor detection.
<p align="center">
  <img src="precision.png" width="280"> 
  <img src="recall.png" width="280">
  <img src="f1_score.jpg" width="280">
</p>

---

## 💻 Clinical Interface & Segmentation Results
The NeuroScan AI dashboard provides a seamless experience for radiologists to upload and analyze MRI scans in real-time.

### 🖥️ User Interface Dashboard
<p align="center">
  <img src="website_image.jpg" width="900" alt="Clinical Dashboard">
</p>

### 🧩 Precision Segmentation & Severity Assessment
By integrating **SAM2 (Segment Anything Model)** with **YOLOv11**, the system achieves pixel-level boundary tracing. This allows for the automated calculation of the **Tumor-to-Brain Area Ratio**, providing a quantitative severity metric.
<p align="center">
  <img src="segmentation_result.jpg" width="900" alt="Segmentation Output">
</p>

---

## 🛠️ Technology Stack
* **Classification:** VGG16 (Transfer Learning)
* **Object Detection:** YOLOv11
* **Segmentation:** SAM2 (Meta AI)
* **Frameworks:** Streamlit, TensorFlow, PyTorch, OpenCV

---

## 📂 Project Structure & Requirements
To execute the suite locally, ensure the following model weights are present in the root directory:
* `final_weights.weights.h5` (VGG16 Weights)
* `best (5).pt` (YOLOv11 Weights)
* `sam2_b.pt` (SAM2 Weights)

---

## 🚀 Installation & Usage

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git](https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git)
