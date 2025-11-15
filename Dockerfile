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

# --- The model download lines have been removed ---
# We don't need them because the model is now in Git LFS
# and will be copied by the 'COPY . .' command.

# Copy the rest of the application's code into the container
# This will copy app.py, requirements.txt, and the 'weights' folder
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
# This now correctly points to the 'app' object inside 'app.py'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]