FROM nvcr.io/nvidia/tritonserver:24.05-py3

# Set environment variables for Triton
ENV MODEL_REPOSITORY /models
ENV PYTHONUNBUFFERED TRUE

# Create model repository directory
RUN mkdir -p ${MODEL_REPOSITORY}

# Copy model repository into the container
COPY model_repository/ ${MODEL_REPOSITORY}/

# Install necessary dependencies
RUN pip install opencv-python-headless \
  numpy \
  ultralytics \
  xgboost

# Set the entrypoint for Triton server
ENTRYPOINT ["tritonserver", "--model-repository=${MODEL_REPOSITORY}"]
