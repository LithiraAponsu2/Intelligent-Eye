# Use the official Python image with OpenGL support
FROM python:3.10-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and other necessary files to the container
COPY . .

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
RUN pip3 install ultralytics opencv-python-headless numpy joblib xgboost Flask

# Set the entrypoint to run the Flask app
ENTRYPOINT ["python3", "docker_flask.py"]
