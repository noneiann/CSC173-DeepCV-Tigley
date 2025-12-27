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
NUM_SYNTHETIC_IMAGES = 1000
MIN_COINS, MAX_COINS = 1, 10
MIN_SCALE, MAX_SCALE = 0.1, 0.25
ALLOW_STACKING = True
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

# Philippine coin diameters (mm) for relative sizing
# 0=1c, 1=5c, 2=10c, 3=25c, 4=1p, 5=5p, 6=10p, 7=20p
COIN_DIAMETERS_MM = {
    0: 15.0,   # 1 centavo
    1: 16.5,   # 5 centavo
    2: 17.0,   # 10 centavo
    3: 20.0,   # 25 centavo
    4: 24.0,   # 1 piso
    5: 25.0,   # 5 piso
    6: 26.5,   # 10 piso
    7: 27.0,   # 20 piso
}
MAX_DIAMETER = max(COIN_DIAMETERS_MM.values())
COIN_RELATIVE_SIZES = {k: v / MAX_DIAMETER for k, v in COIN_DIAMETERS_MM.items()}

# Class names for YOLO (index = class ID)
CLASS_NAMES = [
    "1centavo",   # 0
    "5centavo",   # 1
    "10centavo",  # 2
    "25centavo",  # 3
    "1piso",      # 4
    "5piso",      # 5
    "10piso",     # 6
    "20piso",     # 7
]

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
    
    coin_class = get_coin_class(coin_path)
    if coin_class is None:
        return None, None
    
    relative_size = COIN_RELATIVE_SIZES.get(coin_class, 1.0)
    
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
coins = [p for p in coins_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
print(f"Found {len(coins)} coins, {len(backgrounds)} backgrounds")

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
    
    for _ in range(num_coins):
        coin_path = random.choice(coins)
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
    
    # Save to appropriate split
    split = get_split(i, NUM_SYNTHETIC_IMAGES)
    img_name = f"synthetic_{i:04d}"
    
    cv2.imwrite(str(output_dir / "images" / split / f"{img_name}.jpg"), result)
    with open(output_dir / "labels" / split / f"{img_name}.txt", "w") as f:
        f.write("\n".join(annotations))
    
    print(f"[{split}] {img_name}.jpg - {len(annotations)} coins")

# Create data.yaml for YOLO training
data_yaml = {
    "path": str(output_dir.resolve()),
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