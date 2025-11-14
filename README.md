---
title: Vehicle Tracking
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


# Multi-Vehicle Tracking using the SORT Algorithm

This project is a Python-based implementation of the **SORT (Simple Online and Realtime Tracking)** algorithm. It uses YOLOv8 for object detection and a Kalman Filter for state estimation to track multiple vehicles in a video stream.

The entire application is containerized with Docker and exposed as a FastAPI endpoint, ready for deployment on platforms like Hugging Face Spaces.

[![Deploy to Hugging Face Spaces](https://img.shields.io/badge/🤗%20Deploy-to%20Spaces-blue.svg)](https://huggingface.co/spaces/new?template=fastapi)

---

### How It Works

The tracking logic follows the core principles of the SORT algorithm:

1.  **Detection:** A pre-trained **YOLOv8n** model scans each frame to detect vehicles (cars, trucks, buses).
2.  **Prediction:** For each existing track, a **Kalman Filter** (using a constant velocity model) predicts its new position in the current frame.
3.  **Association:** The predicted bounding boxes (from tracks) are matched with the newly detected bounding boxes (from YOLO). This is solved as an assignment problem using the **Hungarian algorithm**, with Intersection over Union (IoU) as the cost metric.
4.  **Track Management:**
    * Matched detections are used to update the corresponding Kalman Filters.
    * Unmatched detections are used to create new tracks.
    * Unmatched tracks are marked as "unseen." Tracks that are unseen for too many consecutive frames are deleted.

---

### API Endpoint

The server provides one main endpoint for video processing.

**POST `/track_video/`**

* **Request:** `multipart/form-data` containing a video file.
    * `file`: The video file to be processed.
* **Response:** The processed video (with tracking boxes and IDs) as a `video/mp4` file.

---

### How to Use

#### 1. Using the API (cURL)

Once deployed, you can send a video for processing using `curl`:

```bash
curl -X POST "[http://your-app-url.com/track_video/](http://your-app-url.com/track_video/)" \
     -F "file=@/path/to/your_test_video.mp4" \
     -o "output_video.mp4"

echo "Processing complete. Video saved to output_video.mp4"