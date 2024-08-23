# Start from Python 3.10 base image with GPU support
FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3.10 \
  python3.10-dev \
  python3-pip \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Install GPU-specific libraries and dependencies
RUN apt-get update && apt-get install -y \
  nvidia-cuda-toolkit \
  && rm -rf /var/lib/apt/lists/*

# Install Ultralytics library
RUN pip install opencv-python-headless numpy ultralytics joblib xgboost

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application code into the container
COPY . .

# Command to run the Python script (adjust as per your script's name and arguments)
CMD ["python3", "roi_direct_no_show.py"]
