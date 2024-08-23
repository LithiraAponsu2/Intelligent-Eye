#!/bin/bash

# Set the model repository path
MODEL_REPOSITORY=/model

# Start Triton server
tritonserver --model-repository=$MODEL_REPOSITORY