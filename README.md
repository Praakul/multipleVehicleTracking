---
title: Vehicle Tracking
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Multi-Vehicle Tracking using the SORT Algorithm

This project is a Python-based implementation of the SORT (Simple Online and Realtime Tracking) algorithm. It uses YOLOv8 for object detection and a Kalman Filter for state estimation to track multiple vehicles in a video stream.

The entire application is containerized with Docker and deployed as a FastAPI API on Hugging Face Spaces.

## How It Works

The tracking logic follows the core principles of the SORT algorithm:

1. **Detection**: A pre-trained YOLOv8n model scans each frame to detect vehicles (cars, trucks, buses).

2. **Prediction**: For each existing track, a Kalman Filter (using a constant velocity model) predicts its new position in the current frame.

3. **Association**: The predicted bounding boxes (from tracks) are matched with the newly detected bounding boxes (from YOLO). This is solved as an assignment problem using the Hungarian algorithm, with Intersection over Union (IoU) as the cost metric.

4. **Track Management**:
   * Matched detections are used to update the corresponding Kalman Filters.
   * Unmatched detections are used to create new tracks.
   * Unmatched tracks are marked as "unseen" and are deleted if they are not re-detected within a set number of frames.

## How to Use the Live API

This application is an API, not a website. The easiest way to use it is through the built-in documentation page.

### 1. hf page (Recommended)
1. click on "https://huggingface.co/spaces/PrajwalKulkarni/vehicleTracking" 
2. upload video and then wait until it returns a video on the screen which can be viewed and downloaded.


### 2. Using `curl` (Command Line)

You can also use the API programmatically from your terminal.
```bash
curl -X POST "https://prajwalkulkarni-vehicletracking.hf.space/track_video/" \
     -F "file=@/path/to/your_local_video.mp4" \
     -o "tracked_video_output.mp4"
```

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/Praakul/multipleVehicleTracking.git
cd multipleVehicleTracking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web server:
```bash
uvicorn app:app --reload --port 7860
```

4. Access the local documentation: Open your browser and go to `http://127.0.0.1:7860/docs` to use the API.