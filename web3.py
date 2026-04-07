import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
import cv2
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
    
    /* Info Card for Research */
    .info-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4E9F9F;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------- 🧠 MODEL ENGINE (Unchanged Logic) -----------------
@st.cache_resource
def load_all_models():
    yolo = YOLO(r"C:\Users\lak81\OneDrive\Desktop\sam2_yolo11\best (5).pt")
    sam = SAM(r"C:\Users\lak81\OneDrive\Desktop\sam2_yolo11\sam2_b.pt")
    base_model = VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))
    cls_model = Sequential([
        base_model, Flatten(),
        Dense(128, activation='relu', name='dense'), 
        Dropout(0.5),
        Dense(4, activation='softmax', name='dense_1')
    ])
    cls_model.load_weights(r"C:\Users\lak81\OneDrive\Desktop\classification\final_weights.weights.h5")
    return yolo, sam, cls_model

yolo_model, sam_model, cls_model = load_all_models()

def process_scan(image_input):
    img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    yolo_results = yolo_model(img_bgr, verbose=False)
    img_cls = cv2.resize(np.array(image_input), (128, 128)).astype('float32') / 255.0
    cls_preds = cls_model.predict(np.expand_dims(img_cls, 0), verbose=False)
    labels = ['GLIOMA', 'MENINGIOMA', 'NO TUMOR', 'PITUITARY']
    t_type = labels[np.argmax(cls_preds)]
    c_conf = np.max(cls_preds)

    if not yolo_results[0].boxes or t_type == "NO TUMOR":
        return None, 0.0, 1.0, "NO TUMOR", c_conf
    
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    det_conf = yolo_results[0].boxes.conf[0].cpu().item()
    sam_results = sam_model(img_bgr, bboxes=boxes, verbose=False)
    mask = sam_results[0].masks.data[0].cpu().numpy()
    area_pct = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
    mask_resized = cv2.resize(mask.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]))
    overlay = img_bgr.copy()
    overlay[mask_resized > 0] = [242, 95, 92]
    final_img = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
    return cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), area_pct, det_conf, t_type, c_conf

# ----------------- 🛠️ SIDEBAR (Severity Info Added) -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491413.png", width=80)
    st.header("Clinical Dashboard")
    st.write("🏥 **Hospital:** AIIMS Delhi")
    st.markdown("---")
    
    # Severity Ratio Guide in Sidebar
    st.subheader("📊 Severity Scale Guide")
    st.info("""
    **Ratio (Area %):**
    - **0%:** Safe / Healthy
    - **0.1% - 2.5%:** Low Risk
    - **2.6% - 8.0%:** Moderate Risk
    - **> 8.0%:** Critical Risk
    """)
    st.warning("⚠️ *AI predictions must be verified by a Radiologist.*")

# ----------------- 🛠️ NAVIGATION -----------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Live Diagnosis", "About Tech"],
    icons=["house", "search", "info-circle"],
    default_index=1,
    orientation="horizontal",
    styles={"nav-link-selected": {"background-color": "#4E9F9F"}}
)

# ----------------- 🏠 HOME PAGE -----------------
if selected == "Home":
    st.markdown('<div class="header-box"><h1>NeuroScan AI Clinical Suite</h1><p>Next-Gen Hybrid Diagnosis for Neuro-Oncology</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Processing Time", "0.4s")
    c2.metric("System Status", "Online")
    c3.metric("Models Loaded", "VGG16+YOLO+SAM")

# ----------------- 🔬 LIVE DIAGNOSIS -----------------
elif selected == "Live Diagnosis":
    st.header("🔬 MRI Pathology Lab")
    uploaded_file = st.file_uploader("Upload MRI Scan (JPEG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        col_l, col_r = st.columns(2)
        with col_l:
            st.image(raw_img, caption="Input MRI Scan", use_container_width=True)
            
        if st.button("EXECUTE ANALYSIS"):
            with st.spinner("AI Engine Analysis..."):
                res_img, area, d_conf, t_type, c_conf = process_scan(raw_img)
                with col_r:
                    if res_img is not None: st.image(res_img, caption="AI Segmented Output", use_container_width=True)
                    else: st.success("No abnormal patterns detected.")

                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("PATHOLOGY TYPE", t_type)
                m2.metric("TUMOR AREA RATIO", f"{area:.2f}%")
                m3.metric("AI CONFIDENCE", f"{c_conf*100:.1f}%")

                if t_type == "NO TUMOR" or area == 0:
                    st.markdown('<div class="badge badge-safe">✅ STATUS: SAFE - Healthy Tissue</div>', unsafe_allow_html=True)
                elif area > 8.0:
                    st.markdown('<div class="badge badge-critical">🚨 STATUS: CRITICAL - Immediate Intervention Required</div>', unsafe_allow_html=True)
                elif area > 2.5:
                    st.markdown('<div class="badge badge-moderate">⚠️ STATUS: MODERATE - Clinical Follow-up Advised</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge badge-safe">🟢 STATUS: LOW RISK - Monitor Localized Growth</div>', unsafe_allow_html=True)
# -------------------- CLINICAL DISCLAIMER -------------------------
                st.markdown("""<div class="disclaimer-box"><strong>⚠️ Medical Disclaimer:</strong> This automated report is for research purposes based on AIIMS Delhi Research Framework. It does not constitute a formal medical diagnosis. All findings must be reviewed by a board-certified Neuro-Radiologist.</div>""", unsafe_allow_html=True)

# ----------------- 📜 ABOUT TECH (Information Added) -----------------
# ----------------- 📜 ABOUT TECH (Fixed Syntax) -----------------
elif selected == "About Tech":
    st.title("🛡️ Technology & Medical Insights")
    
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Brain Tumor Classification</h3>
            <p>We use <b>VGG16 Transfer Learning</b> to categorize tumors into 4 distinct classes:</p>
            <ul>
                <li><b>Glioma:</b> Originates in the glial cells.</li>
                <li><b>Meningioma:</b> Arises from the meninges (brain membranes).</li>
                <li><b>Pituitary:</b> Abnormal growth in the pituitary gland.</li>
                <li><b>No Tumor:</b> Healthy brain scan.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    
    with t2:
        st.markdown("""
        <div class="info-card">
            <h3>🧩 Smart Segmentation (SAM2)</h3>
            <p>Our <b>SAM2 (Segment Anything Model)</b> is combined with YOLOv11 to provide pixel-level precision.
            It identifies the exact shape of the tumor mass, allowing us to calculate the <b>Severity Ratio</b> based on tissue volume.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("How we calculate Severity?")
    st.write("The severity is determined by the ratio of the tumor pixels to the total pixels in the MRI slice. This gives a volumetric insight into the growth stages.")

st.markdown("<br><hr><center>© 2026 NeuroScan Medical Systems</center>", unsafe_allow_html=True)