from ultralytics import YOLO
import cv2
from pathlib import Path
import pillow_avif  # For AVIF support
from PIL import Image
import numpy as np

model = YOLO("models/v2_model/runs/detect/train/weights/best.pt")

# Image directory - replace with your image folder path
image_dir = "datasets/for testing"
image_path = Path(image_dir)

# Supported formats including AVIF
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp', '*.avif']
image_files = []
for ext in image_extensions:
    image_files.extend(list(image_path.glob(ext)))
    image_files.extend(list(image_path.glob(ext.upper())))

if not image_files:
    print(f"Error: No images found in '{image_dir}'")
    exit()

image_files = sorted(image_files)
print(f"Found {len(image_files)} images. Use 'n' for next, 'p' for previous, 'q' to quit.")

current_index = 0

while True:
    # Load current image
    img_path = image_files[current_index]
    print(f"\nShowing [{current_index + 1}/{len(image_files)}]: {img_path.name}")
    
    # Handle AVIF format with PIL, convert to OpenCV format
    if img_path.suffix.lower() == '.avif':
        pil_img = Image.open(img_path)
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        frame = cv2.imread(str(img_path))
    
    if frame is None:
        print(f"Error: Could not load image '{img_path}'")
        current_index = (current_index + 1) % len(image_files)
        continue
    
    # Run YOLO model on the image
    results = model(frame)
    
    # Process results and draw bounding boxes
    for result in results:
        # Visualize results
        annotated_frame = result.plot()
        
        # Display the resulting frame
        cv2.imshow('YOLO Object Detection - Images', annotated_frame)
    
    # Wait for key press
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q'):
        # Quit
        break
    elif key == ord('n') or key == 83:  # 'n' or right arrow
        # Next image
        current_index = (current_index + 1) % len(image_files)
    elif key == ord('p') or key == 81:  # 'p' or left arrow
        # Previous image
        current_index = (current_index - 1) % len(image_files)

cv2.destroyAllWindows()
print("Application closed.")
