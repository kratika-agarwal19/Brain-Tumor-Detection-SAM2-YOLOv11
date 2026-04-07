from ultralytics import YOLO, SAM
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ---------------- LOAD MODELS ----------------
yolo_model = YOLO(r"C:\Users\lak81\OneDrive\Desktop\sam2_yolo11\best (5).pt")
sam_model = SAM(r"C:\Users\lak81\OneDrive\Desktop\sam2_yolo11\sam2_b.pt")

# ---------------- INPUT IMAGE ----------------
image_path = "C:\\Users\\lak81\\OneDrive\\Desktop\\sam2_yolo11\\test_images\\glioma_1088_jpg.rf.5542c8b3dc2add56cd7303d7007e3ae8.jpg"   # <- change image here
img = cv2.imread(image_path)

# ---------------- YOLO DETECTION ----------------
yolo_results = yolo_model(image_path)

if yolo_results[0].boxes is None:
    print("No tumor detected ❌")
    exit()

boxes = yolo_results[0].boxes.xyxy.cpu().numpy()

# ---------------- SAM SEGMENTATION ----------------
sam_results = sam_model(image_path, bboxes=boxes)

mask = sam_results[0].masks.data[0].cpu().numpy()

# ---------------- AREA CALCULATION ----------------
tumor_pixels = np.sum(mask > 0)
total_pixels = mask.shape[0] * mask.shape[1]

area_percentage = (tumor_pixels / total_pixels) * 100

# ---------------- SEVERITY ----------------
if area_percentage < 5:
    severity = "Low"
elif area_percentage < 15:
    severity = "Medium"
else:
    severity = "High"

print(f"Tumor Area: {area_percentage:.2f}%")
print(f"Severity: {severity}")

# ---------------- OVERLAY MASK ----------------
mask_resized = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]))

overlay = img.copy()
overlay[mask_resized > 0] = [0, 0, 255]  # Red tumor area

# ---------------- SAVE IMAGE ----------------
os.makedirs("results", exist_ok=True)
save_path = "results/output.jpg"
cv2.imwrite(save_path, overlay)

print("Saved at:", save_path)

# ---------------- SHOW IMAGE ----------------
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Area: {area_percentage:.2f}% | Severity: {severity}")
plt.axis('off')
plt.show()
