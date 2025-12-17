import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

# Configuration
SOURCE_DIR = "datasets/human detection dataset/1"
OUTPUT_DIR = "datasets/yolo_annotations"
CONFIDENCE_THRESHOLD = 0.25

# Load YOLOv8 model (or latest available)
# Note: YOLOv12 doesn't exist yet. Using YOLOv8 (can change to yolov11n.pt if available)
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')  # nano model for speed, change to yolov8s.pt or yolov11n.pt if needed
print("Model loaded!")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_and_annotate(img_path, output_label_path):
    """Detect person in image and create YOLO format annotation"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load: {img_path}")
        return False
    
    h_img, w_img = img.shape[:2]
    
    # Run YOLO detection
    results = model(img, verbose=False)
    
    # Filter for person class (class 0 in COCO dataset)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Class 0 is 'person' in COCO
            if cls == 0 and conf >= CONFIDENCE_THRESHOLD:
                # Get bbox in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = ((x1 + x2) / 2) / w_img
                y_center = ((y1 + y2) / 2) / h_img
                width = (x2 - x1) / w_img
                height = (y2 - y1) / h_img
                
                detections.append((0, x_center, y_center, width, height, conf))
    
    # Save annotations
    if detections:
        with open(output_label_path, 'w') as f:
            for det in detections:
                cls_id, xc, yc, w, h, conf = det
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        print(f"✓ {img_path.name}: {len(detections)} person(s) detected")
        return True
    else:
        print(f"✗ {img_path.name}: No person detected")
        return False

# Process all images
import os
source_path = Path(SOURCE_DIR)
if not source_path.exists():
    print(f"ERROR: Source directory '{SOURCE_DIR}' does not exist!")
    exit(1)

# Find all images
image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.jpeg"))
print(f"\nFound {len(image_files)} images to annotate\n")

if len(image_files) == 0:
    print("ERROR: No images found!")
    exit(1)

# Process each image
successful = 0
failed = 0

for img_file in image_files:
    output_label = Path(OUTPUT_DIR) / (img_file.stem + ".txt")
    if detect_and_annotate(img_file, output_label):
        successful += 1
    else:
        failed += 1

print(f"\n{'='*50}")
print(f"Annotation complete!")
print(f"Successfully annotated: {successful}")
print(f"No detections: {failed}")
print(f"Annotations saved to: {OUTPUT_DIR}")
print(f"{'='*50}")
