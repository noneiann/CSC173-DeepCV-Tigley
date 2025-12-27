"""
Capture real coin images from webcam for training data.
This will help close the synthetic-to-real gap.

Instructions:
1. Place coins on a surface
2. Press 1-4 to label what's in the frame:
   1 = 1 peso only
   5 = 5 pesos only  
   0 = 10 pesos only
   2 = 20 pesos only
   m = mixed (will need manual annotation later)
3. Press 's' to save current frame
4. Press 'q' to quit

After capturing, upload to Roboflow or use LabelImg to annotate bounding boxes.
"""

import cv2
import os
from datetime import datetime

# Output directory
OUTPUT_DIR = "datasets/real_captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track captures
capture_count = 0
current_label = "unlabeled"

def main():
    global capture_count, current_label
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("=" * 50)
    print("COIN TRAINING DATA CAPTURE")
    print("=" * 50)
    print("Place coins on surface and capture images")
    print("")
    print("CONTROLS:")
    print("  1 = Label as 1 peso")
    print("  5 = Label as 5 pesos")
    print("  0 = Label as 10 pesos")
    print("  2 = Label as 20 pesos")
    print("  m = Label as mixed (manual annotation needed)")
    print("  s = Save current frame")
    print("  q = Quit")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw info panel
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Current label: {current_label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Captures: {capture_count}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 1/5/0/2/m to set label, 's' to save", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Capture Training Data", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_label = "1piso"
            print(f"Label set to: {current_label}")
        elif key == ord('5'):
            current_label = "5piso"
            print(f"Label set to: {current_label}")
        elif key == ord('0'):
            current_label = "10piso"
            print(f"Label set to: {current_label}")
        elif key == ord('2'):
            current_label = "20piso"
            print(f"Label set to: {current_label}")
        elif key == ord('m'):
            current_label = "mixed"
            print(f"Label set to: {current_label}")
        elif key == ord('s'):
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{current_label}_{timestamp}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Save clean frame (without overlay)
            ret, clean_frame = cap.read()
            if ret:
                cv2.imwrite(filepath, clean_frame)
                capture_count += 1
                print(f"Saved: {filename} (Total: {capture_count})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nDone! Captured {capture_count} images to {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Upload images to Roboflow (https://roboflow.com)")
    print("2. Draw bounding boxes around each coin")
    print("3. Export as YOLO format")
    print("4. Add to your training dataset")

if __name__ == "__main__":
    main()
