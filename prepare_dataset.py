
import cv2
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = "datasets/poses"  # Where you manually sorted images
OUTPUT_DIR = "datasets/processed_thermal"
IMG_SIZE = 640

# Map folder names to YOLO Class IDs
CLASS_MAP = {
    "bending": "bending",
    "lying": "lying",
    "sitting": "sitting",
    "standing": "standing"
}

def create_yolo_label(img_shape, bbox):
    # Convert OpenCV bbox (x, y, w, h) to YOLO (x_center, y_center, w, h) normalized
    h_img, w_img = img_shape[:2]
    x, y, w, h = bbox
    
    x_center = (x + w / 2) / w_img
    y_center = (y + h / 2) / h_img
    width = w / w_img
    height = h / h_img
    
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_image(img_path, output_img_path, output_label_path, class_id):
    # 1. Load as grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return

    # 2. Binarize (Ensure perfect Black/White)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 

    # Find the largest contour (the person)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Check if box is too small (noise)
    if w < 10 or h < 10: return

    # 4. Generate YOLO Label Line
    yolo_coords = create_yolo_label(img.shape, (x, y, w, h))
    label_line = f"{class_id} {yolo_coords}\n"

    # 5. Create "Pseudo-Thermal" Visual

    #resize to standard
    resized_binary = cv2.resize(binary, (IMG_SIZE, IMG_SIZE))
    
    # Apply Blur (Simulate heat dispersion)
    blurred = cv2.GaussianBlur(resized_binary, (21, 21), 0)
    
    # Create natural heat gradient instead of random noise
    # Generate smooth Perlin-like noise using multiple scales
    h, w = blurred.shape
    
    # Create base gradient (warmer at core, cooler at extremities)
    y_coords, x_coords = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    # Invert: 0 at edges (cold), 1 at center (hot)
    radial_gradient = 1 - (distance_from_center / max_dist)
    
    # Add smooth multi-scale variation (simulates uneven body heat distribution)
    gradient_1 = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (71, 71), 0)  
    gradient_2 = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (41, 41), 0)
    gradient_3 = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (21, 21), 0)

    
    # Combine gradients with different weights
    heat_variation = (gradient_1 * 0.08 + gradient_2 * 0.05 + gradient_3 * 0.02)
    heat_variation = (heat_variation - heat_variation.min()) / (heat_variation.max() - heat_variation.min())
    
    # Combine radial gradient with heat variation
    natural_heat = radial_gradient * 0.75 + heat_variation * 0.25
    
    # Normalize heat map to use full 0-255 range
    natural_heat = (natural_heat - natural_heat.min()) / (natural_heat.max() - natural_heat.min())
    
    # Apply heat gradient only where the person is (mask with blurred silhouette)
    mask = (blurred > 0).astype(np.float32)
    background_temp = np.full((h, w), 30, dtype=np.uint8)  # Low ambient heat

    final_img = np.where(mask > 0, (natural_heat * mask * 255).astype(np.uint8), background_temp)
    
    # Apply False Color (Thermal effect) - COLORMAP_JET uses blue->green->yellow->red
    # Blue/Green = cold, Yellow/Red = hot
    thermal_img = cv2.applyColorMap(final_img, cv2.COLORMAP_JET)

    # 6. Save Files
    cv2.imwrite(str(output_img_path), thermal_img)
    with open(output_label_path, 'w') as f:
        f.write(label_line)
    
    print(f"Processed: {img_path.name} -> Class {class_id}")
# --- MAIN EXECUTION ---
# Setup directories
for split in ['train', 'val']:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# Process
images_processed = 0
for folder, class_id in CLASS_MAP.items():
    folder_path = Path(SOURCE_DIR) / folder
    files = list(folder_path.glob("*.*")) # Grab all images
    
    # Simple 80/20 Train/Val split
    split_idx = int(len(files) * 0.8)
    
    for i, file in enumerate(files):
        # Decide if train or val
        subset = "train" if i < split_idx else "val"
        
        # Define output filenames
        filename = f"{folder}_{file.stem}" # Unique name
        out_img = Path(OUTPUT_DIR) / "images" / subset / (filename + ".jpg")
        out_lbl = Path(OUTPUT_DIR) / "labels" / subset / (filename + ".txt")
        
        process_image(file, out_img, out_lbl, class_id)
        images_processed += 1

print(f"Done! Processed {images_processed} images.")