import cv2
import numpy as np
from pathlib import Path
import random
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Paths
coins_dir = Path("datasets/coins")
background_dir = Path("datasets/dtd/images")
output_dir = Path("datasets/synthetic")

# Create YOLO directory structure
for split in ["train", "val", "test"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Configuration
NUM_SYNTHETIC_IMAGES = 2000  # Reduced - real data from Roboflow is more valuable
MIN_COINS, MAX_COINS = 2, 8  # At least 2 coins to learn relative size
MIN_SCALE, MAX_SCALE = 0.12, 0.22  # Tighter range for more consistent sizing
ALLOW_STACKING = True
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

# Philippine coin diameters (mm) for relative sizing
# Only peso coins: file prefix 4=1p, 5=5p, 6=10p, 7=20p
# Maps file prefix -> new class ID
FILE_TO_CLASS = {
    4: 0,   # 1 piso -> class 0
    5: 1,   # 5 piso -> class 1
    6: 2,   # 10 piso -> class 2
    7: 3,   # 20 piso -> class 3
}

# Keep REALISTIC relative sizes - model must learn appearance, not size
# Size alone is unreliable due to varying camera distance
COIN_DIAMETERS_MM = {
    4: 24.0,   # 1 piso
    5: 25.0,   # 5 piso
    6: 26.5,   # 10 piso
    7: 27.0,   # 20 piso
}
MAX_DIAMETER = max(COIN_DIAMETERS_MM.values())
COIN_RELATIVE_SIZES = {k: v / MAX_DIAMETER for k, v in COIN_DIAMETERS_MM.items()}

# Class names for YOLO (index = class ID)
CLASS_NAMES = [
    "1piso",      # 0
    "5piso",      # 1
    "10piso",     # 2
    "20piso",     # 3
]

# =============================================================================
# AUGMENTATION FUNCTIONS (Simplified & Focused)
# =============================================================================

def augment_color_temperature(image):
    """Simulate warm (indoor/tungsten) and cool (daylight/LED) lighting."""
    if random.random() > 0.5:
        return image
    
    result = image.astype(np.float32)
    temp = random.choice(['warm', 'cool'])
    strength = random.uniform(0.05, 0.15)  # Subtle shift
    
    if temp == 'warm':
        result[:, :, 2] = np.clip(result[:, :, 2] * (1 + strength), 0, 255)  # More red
        result[:, :, 0] = np.clip(result[:, :, 0] * (1 - strength), 0, 255)  # Less blue
    else:
        result[:, :, 0] = np.clip(result[:, :, 0] * (1 + strength), 0, 255)  # More blue
        result[:, :, 2] = np.clip(result[:, :, 2] * (1 - strength), 0, 255)  # Less red
    
    return result.astype(np.uint8)

def augment_brightness(image):
    """Random brightness adjustment - simulates different lighting intensity."""
    if random.random() > 0.5:
        return image
    factor = random.uniform(0.7, 1.3)
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def augment_noise(image):
    """Add slight Gaussian noise - simulates camera sensor noise."""
    if random.random() > 0.3:
        return image
    noise = np.random.normal(0, random.uniform(3, 8), image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def augment_blur(image):
    """Slight Gaussian blur - simulates slight camera focus issues."""
    if random.random() > 0.3:
        return image
    return cv2.GaussianBlur(image, (3, 3), 0)

def augment_flip(image, annotations, img_w, img_h):
    """Random horizontal flip only (vertical doesn't make sense for coins on table)."""
    if random.random() > 0.5:
        return image, annotations
    
    image = cv2.flip(image, 1)
    new_annotations = []
    for ann in annotations:
        parts = ann.split()
        cls, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x = 1.0 - x
        new_annotations.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    
    return image, new_annotations

def augment_shear(image, annotations, max_shear=0.05):
    """Very slight shear - simulates slight camera angle."""
    if random.random() > 0.3:
        return image, annotations
    
    h, w = image.shape[:2]
    shear_x = random.uniform(-max_shear, max_shear)
    shear_y = random.uniform(-max_shear, max_shear)
    
    M = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
    
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    new_corners = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    
    min_x, min_y = new_corners.min(axis=0)
    max_x, max_y = new_corners.max(axis=0)
    
    M[0, 2] = -min_x
    M[1, 2] = -min_y
    new_w, new_h = int(max_x - min_x), int(max_y - min_y)
    
    image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(128, 128, 128))
    
    new_annotations = []
    for ann in annotations:
        parts = ann.split()
        cls = int(parts[0])
        x_c, y_c, bw, bh = float(parts[1]) * w, float(parts[2]) * h, float(parts[3]) * w, float(parts[4]) * h
        
        pt = np.array([[[x_c, y_c]]], dtype=np.float32)
        new_pt = cv2.transform(pt, M)[0, 0]
        
        new_x = np.clip(new_pt[0] / new_w, 0, 1)
        new_y = np.clip(new_pt[1] / new_h, 0, 1)
        new_bw = np.clip(bw / new_w, 0, 1)
        new_bh = np.clip(bh / new_h, 0, 1)
        
        new_annotations.append(f"{cls} {new_x:.6f} {new_y:.6f} {new_bw:.6f} {new_bh:.6f}")
    
    return image, new_annotations

def augment_coin_appearance(coin_img):
    """Apply coin-specific augmentations - simulates coin wear/age/lighting."""
    if coin_img is None:
        return coin_img
    
    if coin_img.shape[2] == 4:
        alpha = coin_img[:, :, 3:4]
        bgr = coin_img[:, :, :3]
    else:
        alpha = None
        bgr = coin_img
    
    # Slight color tint (oxidation/wear)
    if random.random() > 0.5:
        tint = np.array([random.uniform(0.95, 1.05), 
                         random.uniform(0.95, 1.05), 
                         random.uniform(0.95, 1.05)])
        bgr = np.clip(bgr * tint, 0, 255).astype(np.uint8)
    
    # Slight brightness variation
    if random.random() > 0.5:
        factor = random.uniform(0.85, 1.15)
        bgr = np.clip(bgr * factor, 0, 255).astype(np.uint8)
    
    if alpha is not None:
        return np.concatenate([bgr, alpha], axis=2)
    return bgr

def apply_augmentations(image, annotations):
    """Apply focused augmentations to image."""
    h, w = image.shape[:2]
    
    # Color/lighting augmentations
    image = augment_color_temperature(image)
    image = augment_brightness(image)
    image = augment_noise(image)
    image = augment_blur(image)
    
    # Geometric augmentations (minimal)
    image, annotations = augment_flip(image, annotations, w, h)
    image, annotations = augment_shear(image, annotations, max_shear=0.05)
    
    return image, annotations

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def overlay_image(background, foreground, x, y):
    """Overlay foreground with alpha onto background at (x, y)."""
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + fg_w), min(bg_h, y + fg_h)
    fg_x1, fg_y1 = x1 - x, y1 - y
    fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return background
    
    roi = background[y1:y2, x1:x2]
    fg_region = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
    
    if fg_region.shape[2] == 4:
        alpha = fg_region[:, :, 3:4] / 255.0
        blended = (fg_region[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
        background[y1:y2, x1:x2] = blended
    else:
        background[y1:y2, x1:x2] = fg_region
    
    return background

def get_coin_class(coin_path):
    """Extract coin class (0-7) from filename."""
    try:
        return int(coin_path.stem.split('_')[0])
    except (ValueError, IndexError):
        return None

def load_and_prepare_coin(coin_path, bg_shape, base_scale):
    """Load coin, apply relative sizing and random rotation."""
    coin = cv2.imread(str(coin_path), cv2.IMREAD_UNCHANGED)
    if coin is None:
        return None, None
    
    file_class = get_coin_class(coin_path)
    if file_class is None:
        return None, None
    
    # Skip centavo coins (only use peso coins: 4, 5, 6, 7)
    if file_class not in FILE_TO_CLASS:
        return None, None
    
    # Map file prefix to new class ID
    coin_class = FILE_TO_CLASS[file_class]
    
    relative_size = COIN_RELATIVE_SIZES.get(file_class, 1.0)
    
    # Add alpha channel if missing
    if len(coin.shape) == 2:
        coin = cv2.cvtColor(coin, cv2.COLOR_GRAY2BGRA)
    elif coin.shape[2] == 3:
        coin = cv2.cvtColor(coin, cv2.COLOR_BGR2BGRA)
    
    # Scale coin based on real-world relative size
    target_size = min(bg_shape[0], bg_shape[1]) * base_scale * relative_size
    scale = target_size / max(coin.shape[0], coin.shape[1])
    new_w, new_h = max(1, int(coin.shape[1] * scale)), max(1, int(coin.shape[0] * scale))
    coin = cv2.resize(coin, (new_w, new_h))
    
    # Apply coin-specific augmentations BEFORE rotation
    coin = augment_coin_appearance(coin)
    
    # Random rotation
    angle = random.uniform(0, 360)
    center = (new_w // 2, new_h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(rot_matrix[0, 0]), np.abs(rot_matrix[0, 1])
    new_w_rot = int(new_h * sin + new_w * cos)
    new_h_rot = int(new_h * cos + new_w * sin)
    rot_matrix[0, 2] += (new_w_rot - new_w) / 2
    rot_matrix[1, 2] += (new_h_rot - new_h) / 2
    coin = cv2.warpAffine(coin, rot_matrix, (new_w_rot, new_h_rot),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    return coin, coin_class

def get_split(index, total):
    """Determine train/val/test split for given index."""
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    if index < train_end:
        return "train"
    elif index < val_end:
        return "val"
    return "test"

# Collect images
backgrounds = [p for p in background_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

# Only collect peso coins (files starting with 4_, 5_, 6_, 7_)
def is_peso_coin(path):
    try:
        prefix = int(path.stem.split('_')[0])
        return prefix in FILE_TO_CLASS
    except:
        return False

all_coins = [p for p in coins_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and is_peso_coin(p)]

# Group coins by class for balanced sampling
coins_by_class = {cls_id: [] for cls_id in range(len(CLASS_NAMES))}
for coin_path in all_coins:
    file_class = int(coin_path.stem.split('_')[0])
    cls_id = FILE_TO_CLASS[file_class]
    coins_by_class[cls_id].append(coin_path)

print(f"Coins per class: {[(CLASS_NAMES[i], len(coins_by_class[i])) for i in range(len(CLASS_NAMES))]}")
print(f"Found {len(all_coins)} peso coins, {len(backgrounds)} backgrounds")

def get_balanced_coin():
    """Pick a coin with equal probability per class (balanced sampling)."""
    cls_id = random.randint(0, len(CLASS_NAMES) - 1)
    return random.choice(coins_by_class[cls_id])

def get_diverse_coins(num_coins):
    """Get coins ensuring class diversity - helps model learn size differences."""
    coins = []
    
    # First, try to include at least one of each class if we have enough coins
    if num_coins >= 4:
        # Include one of each class, then fill randomly
        for cls_id in range(len(CLASS_NAMES)):
            coins.append(random.choice(coins_by_class[cls_id]))
        # Fill remaining with random balanced selection
        for _ in range(num_coins - 4):
            coins.append(get_balanced_coin())
    elif num_coins >= 2:
        # For 2-3 coins, ensure they are different classes
        available_classes = list(range(len(CLASS_NAMES)))
        random.shuffle(available_classes)
        for i in range(num_coins):
            cls_id = available_classes[i % len(available_classes)]
            coins.append(random.choice(coins_by_class[cls_id]))
    else:
        # Single coin - random
        coins.append(get_balanced_coin())
    
    random.shuffle(coins)
    return coins

# Generate synthetic images with YOLO annotations
for i in range(NUM_SYNTHETIC_IMAGES):
    bg_path = random.choice(backgrounds)
    background = cv2.imread(str(bg_path))
    if background is None:
        continue
    
    result = background.copy()
    img_h, img_w = result.shape[:2]
    base_scale = random.uniform(MIN_SCALE, MAX_SCALE)
    num_coins = random.randint(MIN_COINS, MAX_COINS)
    
    annotations = []
    placed_coins = []  # (x, y, w, h, coin_class)
    
    # Get diverse coins to ensure model sees different classes together
    coin_paths = get_diverse_coins(num_coins)
    
    for coin_path in coin_paths:
        coin, coin_class = load_and_prepare_coin(coin_path, background.shape, base_scale)
        if coin is None:
            continue
        
        coin_h, coin_w = coin.shape[:2]
        max_x, max_y = max(0, img_w - coin_w), max(0, img_h - coin_h)
        
        # Position: stack near existing coin or random
        if ALLOW_STACKING and placed_coins and random.random() < 0.5:
            prev_x, prev_y, prev_w, prev_h, _ = random.choice(placed_coins)
            # Offset so coins touch/slightly overlap but don't fully cover
            # Use larger offset (60-90% of combined radius) to ensure visibility
            min_offset = int(max(prev_w, prev_h, coin_w, coin_h) * 0.6)
            max_offset = int(max(prev_w, prev_h, coin_w, coin_h) * 0.9)
            offset_x = random.choice([-1, 1]) * random.randint(min_offset, max_offset)
            offset_y = random.choice([-1, 1]) * random.randint(min_offset, max_offset)
            x = max(0, min(max_x, prev_x + offset_x))
            y = max(0, min(max_y, prev_y + offset_y))
        else:
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
        
        # Check overlap with all placed coins - ensure at least 40% visible
        max_overlap_ratio = 0.6  # Max 60% of coin can be hidden
        valid_position = True
        for px, py, pw, ph, _ in placed_coins:
            # Calculate intersection
            ix1, iy1 = max(x, px), max(y, py)
            ix2, iy2 = min(x + coin_w, px + pw), min(y + coin_h, py + ph)
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                coin_area = coin_w * coin_h
                if intersection / coin_area > max_overlap_ratio:
                    valid_position = False
                    break
        
        # If too much overlap, try random position instead
        if not valid_position:
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
        
        result = overlay_image(result, coin, x, y)
        placed_coins.append((x, y, coin_w, coin_h, coin_class))
        
        # YOLO format: class x_center y_center width height (normalized)
        x_center = (x + coin_w / 2) / img_w
        y_center = (y + coin_h / 2) / img_h
        w_norm = coin_w / img_w
        h_norm = coin_h / img_h
        annotations.append(f"{coin_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    # Save to appropriate split (no augmentation - let YOLO handle it during training)
    split = get_split(i, NUM_SYNTHETIC_IMAGES)
    img_name = f"synthetic_{i:04d}"
    
    cv2.imwrite(str(output_dir / "images" / split / f"{img_name}.jpg"), result)
    with open(output_dir / "labels" / split / f"{img_name}.txt", "w") as f:
        f.write("\n".join(annotations))
    
    print(f"[{split}] {img_name}.jpg - {len(annotations)} coins")

# Create data.yaml for YOLO training
# Use absolute path for compatibility with Colab/different environments
data_yaml = {
    "path": str(output_dir.resolve().as_posix()),  # Absolute path with forward slashes
    "train": "images/train",
    "val": "images/val", 
    "test": "images/test",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES
}
with open(output_dir / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\nDone! Dataset saved to {output_dir}")
print(f"Classes: {CLASS_NAMES}")
print(f"Train: {int(NUM_SYNTHETIC_IMAGES * TRAIN_RATIO)}, Val: {int(NUM_SYNTHETIC_IMAGES * VAL_RATIO)}, Test: {int(NUM_SYNTHETIC_IMAGES * TEST_RATIO)}")