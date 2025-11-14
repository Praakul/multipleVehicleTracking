# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y ffmpeg libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# --- Download the YOLO model during the build ---
# This makes the container self-contained and avoids
# downloading the model every time the app starts.
RUN mkdir -p /app/weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='pt',_check=False); import shutil; shutil.move('yolov8n.pt', '/app/weights/yolov8n.pt')"

# Copy the rest of the application's code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
# This now correctly points to the 'app' object inside 'app.py'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]