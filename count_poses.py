from pathlib import Path
from collections import Counter

# Count poses from annotation files
annotation_dir = Path("datasets/pose_annotations")
pose_counts = Counter()

for txt_file in annotation_dir.glob("*.txt"):
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                pose_class = parts[0]
                pose_counts[pose_class] += 1

print("="*50)
print("Pose Distribution Summary")
print("="*50)
print(f"Total annotation files: {len(list(annotation_dir.glob('*.txt')))}")
print(f"Total detections: {sum(pose_counts.values())}")
print(f"\nBreakdown:")
print(f"  Standing: {pose_counts['standing']}")
print(f"  Sitting: {pose_counts['sitting']}")
print(f"  Lying: {pose_counts['lying']}")
print(f"\nPercentages:")
total = sum(pose_counts.values())
print(f"  Standing: {pose_counts['standing']/total*100:.1f}%")
print(f"  Sitting: {pose_counts['sitting']/total*100:.1f}%")
print(f"  Lying: {pose_counts['lying']/total*100:.1f}%")
print("="*50)
