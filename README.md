🧠 NeuroScan AI: Advanced Neuro-Oncology Suite

An Integrated Pipeline for Brain Tumor Classification & Precision Segmentation

📊 Performance Analytics & Validation

The system is built upon the AIIMS Delhi Research Framework, trained over 50 epochs to ensure high diagnostic reliability.

📈 Training Dynamics (Loss & Accuracy)

The following visualization illustrates the convergence of our hybrid models during the training phase:

<p align="center">
<img src="losses.jpeg" width="850" alt="Training Metrics">
</p>

🔬 Statistical Evaluation

Our models are validated using standard clinical metrics including Precision, Recall, and F1-Score.

<p align="center">
<img src="Precision.jpeg" width="280">
<img src="recall.png" width="280">
<img src="f1%20score.jpg" width="280">
</p>

💻 Clinical Interface & Segmentation Results

The NeuroScan AI dashboard provides a seamless experience for radiologists to analyze MRI scans in real-time.

🖥️ User Interface Dashboard

<p align="center">
<!-- Check if filename is website image.jpg or website_image.jpg -->
<img src="website%20image.jpg" width="900" alt="Clinical Dashboard">
</p>

🧩 Precision Segmentation & Severity Assessment

By integrating SAM2 with YOLOv11, the system achieves pixel-level boundary tracing and automated Tumor-to-Brain Area Ratio calculation.

<p align="center">
<!-- Check if filename has double spaces or special characters -->
<img src="segmentation%20result%20%20with%20severity%20and%20area%20ratio.jpg" width="900" alt="Segmentation Output">
</p>

🛠️ Technology Stack

Classification: VGG16 (Transfer Learning)

Object Detection: YOLOv11

Segmentation: SAM2 (Meta AI)

Framework: Streamlit, TensorFlow, PyTorch

🚀 Installation & Usage

Clone the Repository:

git clone [https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git](https://github.com/kratika-agarwal19/Brain-Tumor-Detection-SAM2-YOLOv11.git)


Install Dependencies:

pip install -r requirements.txt


Run Application:

streamlit run app.py


⚖️ Medical Disclaimer

Disclaimer: This software is a research prototype designed for educational purposes. All AI-generated diagnostic reports must be verified by a board-certified Medical Professional.

<p align="center">© 2026 NeuroScan Medical Analytics | Precision Oncology</p>
