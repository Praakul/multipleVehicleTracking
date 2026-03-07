import cv2
from ultralytics import YOLO
import os
import shutil
from datetime import datetime
from tracker import Tracker
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Vehicle Tracking API")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ultralytics auto-downloads yolov8n.pt if not present
model = YOLO("yolov8n.pt")

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

        if fps == 0:
            print("Warning: Video FPS is 0, setting to 30.")
            fps = 30

        video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        local_tracker = Tracker()
        
        frame_id = 0
        SKIP_FRAMES = 2 # Process every 3rd frame
        last_tracks = [] # Cache the tracks to draw on skipped frames

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_id += 1

            if frame_id % SKIP_FRAMES == 0:
                # Downscale frame for faster inference on CPU
                scale_percent = 50 
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

                # Run inference on smaller frame, half precision helps if supported
                results = model(resized_frame, verbose=False, half=True)
                
                detections = []
                for result in results:
                    for box in result.boxes:
                        # COCO class IDs: 2=car, 5=bus, 7=truck
                        if int(box.cls) in [2, 5, 7]: 
                            # Scale coordinates back up to original size
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            x1 = int(x1 * 100 / scale_percent)
                            y1 = int(y1 * 100 / scale_percent)
                            x2 = int(x2 * 100 / scale_percent)
                            y2 = int(y2 * 100 / scale_percent)
                            
                            detections.append([x1, y1, x2, y2])
                
                last_tracks = local_tracker.update(detections)
            else:
                # On skipped frames, just predict the next state using Kalman
                # without running expensive YOLO inference
                last_tracks = local_tracker.update([])
            
            for track in last_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                track_id = track.track_id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            video_writer.write(frame)

        video_writer.release()
        cap.release()
        print(f"Processing complete. Video saved to: {output_filename}")

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
    
# --- API Endpoints ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the frontend UI."""
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/track_video/")
def track_video_endpoint(file: UploadFile = File(...)):
    """
    Upload a video file, process it for vehicle tracking,
    and return the processed video.
    """
    input_path = os.path.join("uploads", file.filename)
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_filename = os.path.basename(file.filename)
        output_filename = f"outputs/output_{timestamp}_{safe_filename}"

        process_video(input_path, output_filename)

        if not os.path.exists(output_filename):
            return JSONResponse(status_code=500, content={"message": "Failed to process video."})

        return FileResponse(path=output_filename, media_type='video/mp4', filename=os.path.basename(output_filename))
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})
    
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
            print(f"Removed temporary upload file: {input_path}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)