import cv2
from ultralytics import YOLO
import os
import shutil
from datetime import datetime
from tracker import Tracker
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# --- FastAPI App Initialization ---
app = FastAPI(title="Vehicle Tracking API")

# --- Setup and Global Variables ---
# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("weights", exist_ok=True)

model_path = "weights/yolov8n.pt"
# Load the model. It will be downloaded if not present (handled in Dockerfile).
model = YOLO(model_path)

# --- Core Processing Function ---
def process_video(video_path, output_filename):
    """
    Processes a video file to track vehicles and saves the output.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Ensure fps is valid
        if fps == 0:
            print("Warning: Video FPS is 0, setting to 30.")
            fps = 30

        video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Use a fresh tracker for each video processing request
        local_tracker = Tracker()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLOv8 inference, suppressing console output
            results = model(frame, verbose=False)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    # COCO class IDs: 2=car, 5=bus, 7=truck
                    if int(box.cls) in [2, 5, 7]: 
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        detections.append([x1, y1, x2, y2])
            
            # Update the tracker with new detections
            tracks = local_tracker.update(detections)
            
            # Draw tracks on the frame
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                track_id = track.track_id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            video_writer.write(frame)

        video_writer.release()
        cap.release()
        print(f"✅ Processing complete. Video saved to: {output_filename}")

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
    
# --- API Endpoints ---

@app.get("/")
async def root():
    """A simple health check endpoint."""
    return JSONResponse(content={"status": "online", "message": "Vehicle Tracking API is running."})


@app.post("/track_video/")
async def track_video_endpoint(file: UploadFile = File(...)):
    """
    Upload a video file, process it for vehicle tracking,
    and return the processed video.
    """
    input_path = os.path.join("uploads", file.filename)
    
    try:
        # Save the uploaded video file temporarily
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Define the output path
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Ensure the output filename is safe
        safe_filename = os.path.basename(file.filename)
        output_filename = f"outputs/output_{timestamp}_{safe_filename}"

        # Run the core processing logic
        process_video(input_path, output_filename)

        if not os.path.exists(output_filename):
            return JSONResponse(status_code=500, content={"message": "Failed to process video."})

        # Return the processed video file to the user
        return FileResponse(path=output_filename, media_type='video/mp4', filename=os.path.basename(output_filename))
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})
    
    finally:
        # **CRITICAL:** Always delete the temporary uploaded file
        # to prevent the server from running out of disk space.
        if os.path.exists(input_path):
            os.remove(input_path)
            print(f"Removed temporary upload file: {input_path}")

# --- Main execution (for local running) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)