import tritonclient.http as httpclient
import cv2
import numpy as np
import pickle
from collections import defaultdict

# Create a client to communicate with the Triton server
triton_client = httpclient.InferenceServerClient("triton_server_url:8000")

# Initialize track_history
track_history = defaultdict(lambda: [])

# Open the video file
cap = cv2.VideoCapture("show.mp4")

# Define the region of interest with the selected points
roi = np.array([[550, 664], [579, 723], [1120, 697], [998, 646]], np.int32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

# Prepare input data
    inputs = [
        httpclient.InferInput("VIDEO_FRAME", [1, frame.shape[0], frame.shape[1], 3], "FP32"),
        httpclient.InferInput("ROI_ARRAY", roi.shape, "INT32"),
        httpclient.InferInput("TRACK_HISTORY", [1], "BYTES")
    ]
    inputs[0].set_data_from_numpy(frame)
    inputs[1].set_data_from_numpy(roi)
    inputs[2].set_data_from_numpy(np.array([pickle.dumps(track_history)], dtype=np.object_))

    print("frame processed")
    
    # Send the request to Triton and get the prediction
    result = triton_client.infer("your_model_name", inputs)

    # Get the output tensor
    updated_track_history_tensor = result.as_numpy("UPDATED_TRACK_HISTORY")

    # Update the local track_history
    track_history = pickle.loads(updated_track_history_tensor[0])

    # Process the current frame based on the updated track_history
    # ...

# Close the video capture object
cap.release()