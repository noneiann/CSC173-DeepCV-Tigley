import cv2
from ultralytics import YOLO
import numpy as np

# Load your trained model
MODEL_PATH = "models/final model/weights/best.pt"  # Update this path to your trained model

# Class names and colors 
CLASS_NAMES = ["1piso", "5piso", "10piso", "20piso"]
CLASS_VALUES = [1.00, 5.00, 10.00, 20.00]  # Peso values
COLORS = [
    (50, 255, 255),    # 1p - yellow
    (50, 150, 255),    # 5p - orange
    (50, 50, 255),     # 10p - red
    (255, 50, 255),    # 20p - magenta
]

def main():
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded!")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Calculate total value
        total_value = 0.0
        coin_counts = {name: 0 for name in CLASS_NAMES}
        
        # Draw detections
        for box in results[0].boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Get class and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id]
            color = COLORS[cls_id]
            
            # Update counts and total
            coin_counts[cls_name] += 1
            total_value += CLASS_VALUES[cls_id]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw total value panel
        panel_h = 180
        cv2.rectangle(frame, (10, 10), (250, panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, panel_h), (255, 255, 255), 2)
        
        cv2.putText(frame, "COIN COUNTER", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.line(frame, (20, 45), (240, 45), (255, 255, 255), 1)
        
        # Show coin counts (only non-zero)
        y_offset = 65
        for i, (name, count) in enumerate(coin_counts.items()):
            if count > 0:
                text = f"{name}: {count}"
                cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 1)
                y_offset += 20
        
        # Show total
        cv2.line(frame, (20, panel_h - 35), (240, panel_h - 35), (255, 255, 255), 1)
        cv2.putText(frame, f"TOTAL: P{total_value:.2f}", (20, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show help
        cv2.putText(frame, "Press 'q' to quit", (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow("Coin Detector", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("screenshot.jpg", frame)
            print("Screenshot saved!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
