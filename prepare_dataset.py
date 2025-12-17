import cv2
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = "datasets/manual/set 4"  # Where you manually sorted images
OUTPUT_DIR = "datasets/processed_manuals_4"
IMG_SIZE = 640

# Realistic thermal temperature mapping (in Celsius to colormap values)
BODY_CORE_TEMP = 190      # ~36.5°C (torso, head core)
BODY_WARM_TEMP = 170      # ~34°C (upper limbs, upper torso)
BODY_EXTREMITY_TEMP = 140 # ~31°C (hands, feet)
AMBIENT_TEMP = 75         # ~19°C (room temperature)

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
    # 1. Load image
    img = cv2.imread(str(img_path))
    if img is None: return
    
    # 2. Resize to standard size
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 3. Convert to grayscale for luminance-based heat mapping
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 4. Create YOLO label for full image (person occupies most of frame)
    # Assuming person is roughly centered, occupying 70-90% of image
    center_ratio = np.random.uniform(0.75, 0.90)
    yolo_coords = f"0.5 0.5 {center_ratio:.6f} {center_ratio:.6f}"
    label_line = f"{class_id} {yolo_coords}\n"

    # 5. Create "Pseudo-Thermal" Visual - Enhanced with inversion and contrast
    
    # Invert (bright becomes cold, dark becomes hot)
    inverted = 255 - gray
    
    # Contrast stretch (make hot/cold more pronounced)
    normalized = cv2.normalize(inverted, None, 50, 230, cv2.NORM_MINMAX)
    
    # Apply Gaussian blur for heat dispersion effect
    blurred = cv2.GaussianBlur(normalized, (15, 15), 0)
    
    h, w = blurred.shape
    
    # Use the processed thermal intensity
    final_img = blurred.copy()
    
    # Add stronger sensor noise (realistic thermal camera noise)
    sensor_noise = np.random.normal(0, 2.5, (h, w)).astype(np.float32)
    final_img = np.clip(final_img + sensor_noise, 0, 255).astype(np.uint8)
    
    # Add temporal noise (per-pixel random variation)
    temporal_noise = np.random.uniform(-1.5, 1.5, (h, w)).astype(np.float32)
    final_img = np.clip(final_img + temporal_noise, 0, 255).astype(np.uint8)
    
    # Add scan line effects (some thermal cameras have horizontal artifacts)
    if np.random.rand() < 0.4:  # 40% of images
        scan_lines = np.zeros((h, w), dtype=np.float32)
        for row in range(0, h, np.random.randint(20, 40)):
            scan_lines[row:row+1, :] = np.random.uniform(-2, 2)
        final_img = np.clip(final_img + scan_lines, 0, 255).astype(np.uint8)
    
    # Add occasional dead pixels (sensor artifacts)
    if np.random.rand() < 0.4:  # 40% of images have dead pixels
        num_dead_pixels = np.random.randint(3, 12)
        for _ in range(num_dead_pixels):
            px, py = np.random.randint(0, w), np.random.randint(0, h)
            final_img[py:py+2, px:px+2] = np.random.randint(40, 110)
    
    # Simulate lens vignetting (slight darkening at edges)
    y_coords, x_coords = np.ogrid[:h, :w]
    y_normalized = y_coords.astype(np.float32) / h
    x_normalized = x_coords.astype(np.float32) / w
    vignette_mask = np.sqrt((x_normalized - 0.5)**2 + (y_normalized - 0.5)**2)
    vignette_mask = 1 - (vignette_mask / vignette_mask.max()) * 0.15
    final_img = (final_img * vignette_mask).astype(np.uint8)
    
    # Apply False Color (Thermal effect) - COLORMAP_JET uses blue->green->yellow->red
    thermal_img = cv2.applyColorMap(final_img, cv2.COLORMAP_JET)
    
    # Add slight motion blur occasionally (simulates movement/camera shake)
    if np.random.rand() < 0.15:  # 15% of images
        kernel_size = np.random.choice([3, 5])
        thermal_img = cv2.GaussianBlur(thermal_img, (kernel_size, kernel_size), 0)
    
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

# Check if SOURCE_DIR has subfolders or is a flat directory
source_path = Path(SOURCE_DIR)
if not source_path.exists():
    print(f"ERROR: Source directory '{SOURCE_DIR}' does not exist!")
    exit(1)

# Try to find subdirectories matching CLASS_MAP keys
has_subfolders = any((source_path / folder).is_dir() for folder in CLASS_MAP.keys())

if has_subfolders:
    # Process organized folders (bending/lying/sitting/standing)
    print("Processing organized folders...")
    for folder, class_id in CLASS_MAP.items():
        folder_path = source_path / folder
        if not folder_path.exists():
            print(f"Skipping {folder} - folder not found")
            continue
            
        files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
        print(f"Found {len(files)} images in {folder}")
        
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
else:
    # Process flat directory - all images as "standing" class by default
    print(f"Processing flat directory: {SOURCE_DIR}")
    files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.jpeg")) + list(source_path.glob("*.webp")) + list(source_path.glob("*.avif"))
    print(f"Found {len(files)} images")
    
    if len(files) == 0:
        print("ERROR: No images found! Make sure you have .jpg, .png, or .jpeg files")
        exit(1)
    
    # Simple 80/20 Train/Val split
    split_idx = int(len(files) * 0.8)
    class_id = "standing"  # Default class for flat directory
    
    for i, file in enumerate(files):
        # Decide if train or val
        subset = "train" if i < split_idx else "val"
        
        # Define output filenames
        filename = file.stem # Use original filename
        out_img = Path(OUTPUT_DIR) / "images" / subset / (filename + ".jpg")
        out_lbl = Path(OUTPUT_DIR) / "labels" / subset / (filename + ".txt")
        
        process_image(file, out_img, out_lbl, class_id)
        images_processed += 1

print(f"Done! Processed {images_processed} images.")