import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import cv2
import os
import matplotlib.pyplot as plt

# 1. 🏗️ BUILD MODEL ARCHITECTURE
def get_trained_model(weights_path):
    # VGG16 Backbone
    base_model = VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    model.load_weights(weights_path)
    return model

# 2. 🧪 ENHANCED PREPROCESSING (Critical for Unseen Data)
def process_unseen(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None
    
    # Standardizing for medical consistency
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Histogram Equalization to fix brightness mismatch
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Scaling to [0, 1]
    img_resized = cv2.resize(img_enhanced, (128, 128)).astype('float32') / 255.0
    return np.expand_dims(img_resized, axis=0), img_rgb

# 3. 🚀 RUN DIAGNOSTIC TEST
def run_test(folder_path, weights_path):
    model = get_trained_model(weights_path)
    labels = ['GLIOMA', 'MENINGIOMA', 'NO TUMOR', 'PITUITARY']
    
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    num_imgs = len(files)
    
    if num_imgs == 0:
        print("No images found in the folder!")
        return

    # Visualization Setup
    plt.figure(figsize=(15, 5 * ((num_imgs // 3) + 1)))
    
    for i, img_name in enumerate(files):
        full_path = os.path.join(folder_path, img_name)
        input_data, original_img = process_unseen(full_path)
        
        if input_data is not None:
            # Prediction
            preds = model.predict(input_data, verbose=0)[0]
            class_idx = np.argmax(preds)
            confidence = preds[class_idx] * 100
            
            # Formatting Output
            result_text = f"Pred: {labels[class_idx]}\nConf: {confidence:.1f}%"
            color = 'green' if confidence > 80 else 'orange'
            if confidence < 50: color = 'red'
            
            # Plotting
            plt.subplot((num_imgs // 3) + 1, 3, i + 1)
            plt.imshow(original_img)
            plt.title(f"File: {img_name}\n{result_text}", color=color, fontsize=10)
            plt.axis('off')
            
            print(f"✅ Processed: {img_name} -> {labels[class_idx]} ({confidence:.2f}%)")

    plt.tight_layout()
    plt.show()

# --- RUN CONFIG ---
# 1. Weights Path (Make sure this is correct)
MY_WEIGHTS = r"C:\Users\lak81\OneDrive\Desktop\classification\final_weights.weights.h5"

# 2. Folder where you kept unseen images (image(4), image(6), etc.)
MY_TEST_FOLDER = r"C:\Users\lak81\OneDrive\Desktop\classification\test folder"

run_test(MY_TEST_FOLDER, MY_WEIGHTS)