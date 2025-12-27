"""
Combine synthetic + online datasets with class remapping.

Dataset class mappings:
- Synthetic: ["1piso", "5piso", "10piso", "20piso"] (0, 1, 2, 3)
- Online 1: 14 classes (front/back, old/new) -> merge to 4 classes
- Online 2: ["10_PESOS", "1_PESO", "20_PESOS", "5_PESOS"] -> reorder

Target: ["1piso", "5piso", "10piso", "20piso"] (0, 1, 2, 3)
"""

import shutil
from pathlib import Path
import yaml

# Paths
SYNTHETIC_DIR = Path("datasets/synthetic")
ONLINE1_DIR = Path("datasets/online dataset")
ONLINE2_DIR = Path("datasets/online dataset 2")
COMBINED_DIR = Path("datasets/combined")

# Target class names
TARGET_CLASSES = ["1piso", "5piso", "10piso", "20piso"]

# Online Dataset 1: 14 classes -> 4 classes
ONLINE1_REMAP = {
    0: 0, 1: 0, 2: 0, 3: 0,   # 1 peso variants -> 1piso
    4: 2, 5: 2, 6: 2, 7: 2,   # 10 peso variants -> 10piso
    8: 3, 9: 3,               # 20 peso variants -> 20piso
    10: 1, 11: 1, 12: 1, 13: 1,  # 5 peso variants -> 5piso
}

# Online Dataset 2: reorder [10,1,20,5] -> [1,5,10,20]
ONLINE2_REMAP = {0: 2, 1: 0, 2: 3, 3: 1}

def remap_labels(src_path, dst_path, remap_dict):
    with open(src_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_cls = int(parts[0])
            if old_cls in remap_dict:
                new_lines.append(f"{remap_dict[old_cls]} {' '.join(parts[1:])}\n")
    with open(dst_path, 'w') as f:
        f.writelines(new_lines)

def copy_dataset(src_dir, prefix, remap_dict=None):
    if not src_dir.exists():
        print(f"  {prefix}: NOT FOUND")
        return 0
    count = 0
    for split in ["train", "valid", "val", "test"]:
        split_out = "val" if split == "valid" else split
        # Try both folder structures: split/images OR images/split
        img_dir = src_dir / split / "images"
        lbl_dir = src_dir / split / "labels"
        if not img_dir.exists():
            img_dir = src_dir / "images" / split
            lbl_dir = src_dir / "labels" / split
        if not img_dir.exists():
            continue
        imgs = list(img_dir.glob("*.*"))
        imgs = [i for i in imgs if i.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        for idx, img in enumerate(imgs):
            print(f"  [{prefix}] {split_out}: {idx+1}/{len(imgs)} - {img.name}", end="\r")
            shutil.copy(img, COMBINED_DIR / "images" / split_out / f"{prefix}_{img.name}")
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                dst_lbl = COMBINED_DIR / "labels" / split_out / f"{prefix}_{img.stem}.txt"
                if remap_dict:
                    remap_labels(lbl, dst_lbl, remap_dict)
                else:
                    shutil.copy(lbl, dst_lbl)
            count += 1
        if imgs:
            print()  # New line after progress
    print(f"  {prefix}: {count} images")
    return count

def main():
    # Clear old combined
    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)
    for split in ["train", "val", "test"]:
        (COMBINED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (COMBINED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("COMBINING DATASETS")
    print("=" * 50)
    
    total = 0
    total += copy_dataset(SYNTHETIC_DIR, "syn", None)
    total += copy_dataset(ONLINE1_DIR, "on1", ONLINE1_REMAP)
    total += copy_dataset(ONLINE2_DIR, "on2", ONLINE2_REMAP)
    
    # Create data.yaml
    yaml_data = {
        "path": str(COMBINED_DIR.resolve().as_posix()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 4, "names": TARGET_CLASSES
    }
    with open(COMBINED_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f)
    
    print(f"\nTotal: {total} images")
    print(f"Output: {COMBINED_DIR.resolve()}")
    print(f'\nTrain: data="{COMBINED_DIR / "data.yaml"}"')

if __name__ == "__main__":
    main()
