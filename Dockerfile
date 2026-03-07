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
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the YOLOv8n model weights during build
# so they don't need to be shipped in the git repo.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy the rest of the application's code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]