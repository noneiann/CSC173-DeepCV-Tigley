from ultralytics import YOLO
import cv2

model = YOLO("models/v1_model/runs/detect/train/weights/best.pt")

# Load video file - replace with your video path
video_path = "Video_Ready_After_Thermal_Comment.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'.")
    exit()

print(f"Video opened successfully. Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to grab frame")
        break

    # Run YOLO model on the frame for inference
    results = model(frame)

    # Process results and draw bounding boxes on the frame
    for result in results:
        # Get bounding box coordinates and other info
        boxes = result.boxes
        
        # The 'result.plot()' method provides a simple way to visualize results
        annotated_frame = result.plot() 
        
        # Display the resulting frame
        cv2.imshow('YOLO Object Detection - Video', annotated_frame)

    # Break the loop on 'q' key press
    # Adjust waitKey value to control playback speed (1 = fast, 25 = ~normal speed)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()