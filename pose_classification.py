import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

# Configuration
SOURCE_DIR = "datasets/human detection dataset/1"
OUTPUT_DIR = "datasets/pose_annotations"
CONFIDENCE_THRESHOLD = 0.3  # YOLOv11 has better accuracy, can use higher threshold

# Load YOLOv11 Pose model
print("Loading YOLOv11-pose model...")
model = YOLO('YOLO11s-pose.pt')  # Pose estimation model
print("Model loaded!")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COCO Keypoint indices
KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def classify_pose(keypoints):
    """
    Classify pose as standing, sitting, or lying based on keypoint positions
    Uses angle-invariant metrics (joint angles, body segment ratios)
    keypoints: array of shape (17, 3) where each row is [x, y, confidence]
    """
    def get_point(kp):
        """Get (x, y) if confidence > 0.4, else None (lower for YOLOv11)"""
        return (kp[0], kp[1]) if kp[2] > 0.4 else None
    
    def angle_between_points(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3 (in degrees)"""
        if None in [p1, p2, p3]:
            return None
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1.0, 1.0))
        return np.degrees(angle)
    
    def distance(p1, p2):
        """Euclidean distance between two points"""
        if None in [p1, p2]:
            return None
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Extract keypoints
    nose = get_point(keypoints[KEYPOINTS['nose']])
    left_shoulder = get_point(keypoints[KEYPOINTS['left_shoulder']])
    right_shoulder = get_point(keypoints[KEYPOINTS['right_shoulder']])
    left_hip = get_point(keypoints[KEYPOINTS['left_hip']])
    right_hip = get_point(keypoints[KEYPOINTS['right_hip']])
    left_knee = get_point(keypoints[KEYPOINTS['left_knee']])
    right_knee = get_point(keypoints[KEYPOINTS['right_knee']])
    left_ankle = get_point(keypoints[KEYPOINTS['left_ankle']])
    right_ankle = get_point(keypoints[KEYPOINTS['right_ankle']])
    
    # Calculate average points
    shoulder_mid = None
    if left_shoulder and right_shoulder:
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2)
    elif left_shoulder:
        shoulder_mid = left_shoulder
    elif right_shoulder:
        shoulder_mid = right_shoulder
    
    hip_mid = None
    if left_hip and right_hip:
        hip_mid = ((left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2)
    elif left_hip:
        hip_mid = left_hip
    elif right_hip:
        hip_mid = right_hip
    
    # Calculate knee angles (hip-knee-ankle angle)
    left_knee_angle = angle_between_points(left_hip, left_knee, left_ankle)
    right_knee_angle = angle_between_points(right_hip, right_knee, right_ankle)
    
    # Use the best available knee angle
    knee_angle = None
    if left_knee_angle and right_knee_angle:
        knee_angle = (left_knee_angle + right_knee_angle) / 2
    elif left_knee_angle:
        knee_angle = left_knee_angle
    elif right_knee_angle:
        knee_angle = right_knee_angle
    
    # Calculate torso length and leg lengths
    torso_length = distance(shoulder_mid, hip_mid)
    
    left_leg_length = None
    if left_hip and left_ankle:
        left_leg_length = distance(left_hip, left_ankle)
    
    right_leg_length = None
    if right_hip and right_ankle:
        right_leg_length = distance(right_hip, right_ankle)
    
    leg_length = None
    if left_leg_length and right_leg_length:
        leg_length = (left_leg_length + right_leg_length) / 2
    elif left_leg_length:
        leg_length = left_leg_length
    elif right_leg_length:
        leg_length = right_leg_length
    
    # Calculate body aspect ratio (width vs height)
    body_width = None
    body_height = None
    if shoulder_mid and hip_mid:
        body_height = distance(shoulder_mid, hip_mid)
        if left_shoulder and right_shoulder:
            body_width = distance(left_shoulder, right_shoulder)
    
    # CLASSIFICATION LOGIC (fine-tuned for YOLOv11)
    
    # LYING DOWN: Body is wide relative to height (horizontal orientation)
    if body_width and body_height:
        aspect_ratio = body_width / (body_height + 1e-6)
        if aspect_ratio > 1.5:  # Body more horizontal than vertical
            return "lying"
    
    # Priority 1: Use knee angle (most reliable for YOLOv11)
    if knee_angle:
        # STANDING: Legs very extended (>160 degrees - nearly straight)
        if knee_angle > 160:
            return "standing"
        # SITTING: Moderately bent knees (70-140 degrees)
        elif 70 < knee_angle < 140:
            return "sitting"
        # Very bent (< 70) could be lying or crouching, check leg-torso ratio
        elif knee_angle < 70:
            if torso_length and leg_length:
                if leg_length / (torso_length + 1e-6) < 0.7:
                    return "lying"
            return "sitting"  # Default for bent knees
    
    # Priority 2: Use leg-to-torso ratio as fallback
    if torso_length and leg_length:
        leg_to_torso_ratio = leg_length / (torso_length + 1e-6)
        # STANDING: Long extended legs (> 1.4x torso)
        if leg_to_torso_ratio > 1.4:
            return "standing"
        # SITTING: Compact legs (< 1.0x torso)
        elif leg_to_torso_ratio < 1.0:
            return "sitting"
        # Middle range (1.0-1.4): slightly bent, likely sitting
        else:
            return "sitting"
    
    # Default: sitting (most common in dataset)
    return "sitting"

def process_image(img_path, output_label_path):
    """Detect pose and classify, then create YOLO format annotation"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load: {img_path}")
        return None
    
    h_img, w_img = img.shape[:2]
    
    # Run YOLO pose detection
    results = model(img, verbose=False)
    
    # Process detections
    detections = []
    for result in results:
        if result.keypoints is None:
            continue
            
        boxes = result.boxes
        keypoints = result.keypoints
        
        for i, (box, kp) in enumerate(zip(boxes, keypoints)):
            conf = float(box.conf[0])
            
            if conf >= CONFIDENCE_THRESHOLD:
                # Get bbox in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get keypoints (17, 3) - x, y, confidence
                kp_array = kp.data[0].cpu().numpy()
                
                # Classify pose based on keypoints
                pose_class = classify_pose(kp_array)
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / w_img
                y_center = ((y1 + y2) / 2) / h_img
                width = (x2 - x1) / w_img
                height = (y2 - y1) / h_img
                
                detections.append((pose_class, x_center, y_center, width, height, conf))
    
    # Save annotations
    if detections:
        with open(output_label_path, 'w') as f:
            for det in detections:
                pose_class, xc, yc, w, h, conf = det
                f.write(f"{pose_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        
        # Print summary
        class_summary = {}
        for det in detections:
            pose_class = det[0]
            class_summary[pose_class] = class_summary.get(pose_class, 0) + 1
        
        summary_str = ", ".join([f"{count} {cls}" for cls, count in class_summary.items()])
        print(f"✓ {img_path.name}: {summary_str}")
        return detections
    else:
        print(f"✗ {img_path.name}: No person detected")
        return None

# Process all images
source_path = Path(SOURCE_DIR)
if not source_path.exists():
    print(f"ERROR: Source directory '{SOURCE_DIR}' does not exist!")
    exit(1)

# Find all images
image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.jpeg"))
print(f"\nFound {len(image_files)} images to process\n")

if len(image_files) == 0:
    print("ERROR: No images found!")
    exit(1)

# Process each image
successful = 0
failed = 0
pose_counts = {"standing": 0, "sitting": 0, "lying": 0}

for img_file in image_files:
    output_label = Path(OUTPUT_DIR) / (img_file.stem + ".txt")
    detections = process_image(img_file, output_label)
    
    if detections:
        successful += 1
        for det in detections:
            pose_counts[det[0]] = pose_counts.get(det[0], 0) + 1
    else:
        failed += 1

print(f"\n{'='*50}")
print(f"Pose Classification Complete!")
print(f"Successfully processed: {successful}")
print(f"No detections: {failed}")
print(f"\nPose Distribution:")
print(f"  Standing: {pose_counts['standing']}")
print(f"  Sitting: {pose_counts['sitting']}")
print(f"  Lying: {pose_counts['lying']}")
print(f"\nAnnotations saved to: {OUTPUT_DIR}")
print(f"{'='*50}")
