# Start from an existing base image
FROM nvcr.io/nvidia/tritonserver:24.05-py3

# Install required packages
RUN apt-get update && \
  apt-get install -y libgl1-mesa-glx && \
  rm -rf /var/lib/apt/lists/*


