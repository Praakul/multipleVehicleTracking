import cv2
from ultralytics import YOLO
import numpy as np
from tracker import Tracker

# Load the YOLOv8 model
model = YOLO("yolov8n.pt") # Using yolov8n for faster performance

# Open the video file
video_path = "/home/prakul/Desktop/Projects/vehicleTrackingUsingKalman/trafficDatasetKaggle/rouen_video.avi"
cap = cv2.VideoCapture(video_path)

# Initialize our Tracker
tracker = Tracker()

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Extract bounding boxes for 'car', 'truck', 'bus'
    detections = []
    for result in results:
        for box in result.boxes:
            # Check if the detected object is a vehicle (class IDs for car, truck, bus in COCO)
            if int(box.cls) in [2, 5, 7]: 
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append([x1, y1, x2, y2])
    
    # Update the tracker with the new detections
    tracks = tracker.update(detections)
    
    # Draw the tracks on the frame
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        track_id = track.track_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()