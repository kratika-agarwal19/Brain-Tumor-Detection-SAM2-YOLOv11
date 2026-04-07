import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
import cv2
import os
from ultralytics import YOLO, SAM
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# ----------------- 🏥 PAGE CONFIG -----------------
st.set_page_config(
    page_title="NeuroScan AI | Precision Oncology",
    page_icon="🧠",
    layout="wide"
)

# ----------------- 💎 ADVANCED MEDICAL CSS -----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #1B3A57; color: white; }
    .header-box {
        background: linear-gradient(90deg, #1B3A57 0%, #4E9F9F 100%);
        padding: 40px; border-radius: 20px; color: white; text-align: center; margin-bottom: 25px;
    }
    .badge { padding: 8px 20px; border-radius: 25px; font-weight: bold; font-size: 16px; display: inline-block; }
    .badge-critical { background-color: #ff4b4b; color: white; border: 2px solid #b22222; }
    .badge-moderate { background-color: #ffa500; color: white; border: 2px solid #cc8400; }
    .badge-safe { background-color: #28a745; color: white; border: 2px solid #1e7e34; }
    .info-card {
        background-color: #ffffff; padding: 20px; border-radius: 15px;
        border-left: 5px solid #4E9F9F; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .disclaimer-box {
        background-color: #fff3cd; color: #856404; padding: 15px;
        border-radius: 10px; border: 1px solid #ffeeba; margin-top: 20px; font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------- 🧠 MODEL ENGINE (PATHS FIXED) -----------------
@st.cache_resource
def load_all_models():
    # FIXED: Local paths removed for GitHub/Sir's PC compatibility
    yolo_path = "best (5).pt"
    sam_path = "sam2_b.pt"
    vgg_weights = "final_weights.weights.h5"

    try:
        yolo = YOLO(yolo_path)
        sam = SAM(sam_path)
        
        base_model = VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))
        cls_model = Sequential([
            base_model, Flatten(),
            Dense(128, activation='relu', name='dense'), 
            Dropout(0.5),
            Dense(4, activation='softmax', name='dense_1')
        ])
        
        if os.path.exists(vgg_weights):
            cls_model.load_weights(vgg_weights)
        else:
            st.error(f"Missing weights: {vgg_weights}")
            
        return yolo, sam, cls_model
    except Exception as e:
        st.error(f"Error loading models: {e}. Ensure weights are in the main folder.")
        return None, None, None

yolo_model, sam_model, cls_model = load_all_models()

def process_scan(image_input):
    if yolo_model is None: return None, 0.0, 0.0, "SYSTEM ERROR", 0.0
    
    img_array = np.array(image_input)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Classification
    img_cls = cv2.resize(img_array, (128, 128)).astype('float32') / 255.0
    cls_preds = cls_model.predict(np.expand_dims(img_cls, 0), verbose=False)
    labels = ['GLIOMA', 'MENINGIOMA', 'NO TUMOR', 'PITUITARY']
    t_type = labels[np.argmax(cls_preds)]
    c_conf = np.max(cls_preds)

    # Detection
    yolo_results = yolo_model(img_bgr, verbose=False)
    
    if not yolo_results[0].boxes or t_type == "NO TUMOR":
        return None, 0.0, 1.0, "NO TUMOR", c_conf
    
    # Segmentation
    try:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        det_conf = yolo_results[0].boxes.conf[0].cpu().item()
        sam_results = sam_model(img_bgr, bboxes=boxes, verbose=False)
        
        if sam_results[0].masks is not None:
            mask = sam_results[0].masks.data[0].cpu().numpy()
            area_pct = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
            mask_resized = cv2.resize(mask.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]))
            
            overlay = img_bgr.copy()
            overlay[mask_resized > 0] = [242, 95, 92]
            final_img = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
            return cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), area_pct, det_conf, t_type, c_conf
    except:
