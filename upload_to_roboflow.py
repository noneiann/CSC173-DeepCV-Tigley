from roboflow import Roboflow
from pathlib import Path

# Initialize (get API key from roboflow.com/settings)
rf = Roboflow(api_key="7dVmIZHRtVE09mPFsjZH")
project = rf.workspace("museoscholar").project("synthetic-thermal")

# Upload train set
train_imgs = Path("datasets/processed_thermal/images/val")
train_lbls = Path("datasets/yolo_annotations")

for img_file in train_imgs.glob("*.jpg"):
    label_file = train_lbls / (img_file.stem + ".txt")
    if label_file.exists():
        project.upload(
            image_path=str(img_file),
            annotation_path=str(label_file),
            split="train"
        )

# # Upload val set
# val_imgs = Path("datasets/processed_thermal/images/val")
# val_lbls = Path("datasets/processed_thermal/labels/val")

# for img_file in val_imgs.glob("*.jpg"):
#     label_file = val_lbls / (img_file.stem + ".txt")
#     if label_file.exists():
#         project.upload(
#             image_path=str(img_file),
#             annotation_path=str(label_file),
#             split="valid"
#         )

print("Upload complete!")