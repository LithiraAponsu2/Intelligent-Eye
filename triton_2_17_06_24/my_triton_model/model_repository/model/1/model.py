import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import os
from collections import defaultdict

# Load pre-trained models
yolo_model = YOLO("yolov8x-seg.engine")
xgb_model = pickle.load(open("/models/your_model_name/1/finalized_model_xg.sav", 'rb'))  # Adjust the path as necessary

# Define input and output tensors
input_tensors = [
    ("VIDEO_FRAME", np.float32, (1, -1, -1, 3)),
    ("ROI_ARRAY", np.int32, (4, 2)),
    ("TRACK_HISTORY", np.object_, (1,))  # Track history as a serialized object
]
output_tensors = [
    ("UPDATED_TRACK_HISTORY", np.object_, (1,))  # Updated track history as a serialized object
]

# Initialize track_history
track_history = defaultdict(lambda: [])

# Set the save directory (this will be a mounted volume from the host)
save_directory = "/saved_frames"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

def initialize():
    pass

def execute(input_tensors):
    # Unpack input tensors
    video_frame, roi_array, track_history_tensor = input_tensors

    # Convert track_history_tensor to a dictionary
    track_history = pickle.loads(track_history_tensor[0])
    
    # Convert ROI_ARRAY to a numpy array
    roi = np.array(roi_array)
    
    # Calculate roi_binary and area_of_roi
    roi_binary = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.fillPoly(roi_binary, [roi], color=(255, 255, 255))
    area_of_roi = cv2.countNonZero(cv2.cvtColor(roi_binary, cv2.COLOR_BGR2GRAY))
    
    # Calculate line equation from ROI points 1 and 2
    p1, p2 = roi[1], roi[2]
    m, b = calculate_line_equation(p1, p2)

    
    # Run YOLO object detection
    results = yolo_model(video_frame[0])

    # Initialize lists to store detected persons and vehicles
    persons = []
    vehicles = []

    # Identify all persons and bikes/motorcycles
    bike_vehicle_centers = []
    for mask, cls_num in zip(results[0].masks.xy, results[0].boxes.cls.int().cpu().tolist()):
        if mask.size > 0 and yolo_model.names[cls_num] == 'motorcycle':
            center = np.mean(mask, axis=0)
            bike_vehicle_centers.append(center)

    # Extract data for persons and vehicles
    for mask, track_id, confidence, cls_num in zip(results[0].masks.xy, results[0].boxes.id.int().cpu().tolist(), results[0].boxes.conf.float().cpu().tolist(), results[0].boxes.cls.int().cpu().tolist()):
        if mask.size == 0:
            continue

        max_point = max(mask, key=lambda x: (x[0], x[1]))
        if is_to_the_right(max_point, m, b):
            continue  # Skip vehicle passed the crossing upper bound

        center = np.mean(mask, axis=0)

        # Check if this person is on a bike
        if yolo_model.names[cls_num] == 'person' and any(np.linalg.norm(center - bike_center) < 100 for bike_center in bike_vehicle_centers):
            continue  # Skip further processing for this person

        mask_binary = np.zeros_like(video_frame[0])
        cv2.fillPoly(mask_binary, [np.array(mask, dtype=np.int32)], color=(255, 255, 255))
        intersection_binary = cv2.bitwise_and(mask_binary, roi_binary)
        intersection_area = cv2.countNonZero(cv2.cvtColor(intersection_binary, cv2.COLOR_BGR2GRAY))
        mask_area = cv2.countNonZero(cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY))

        # Calculate IoU
        iou = intersection_area / (mask_area + area_of_roi - intersection_area) if mask_area + area_of_roi - intersection_area > 0 else 0

        if yolo_model.names[cls_num] == 'person' and intersection_area > 0:
            persons.append((center, iou * 100))  # IoU percentage
        elif yolo_model.names[cls_num] in ['car', 'motorcycle', 'bus', 'truck'] and intersection_area > 0:
            vehicles.append((center, iou * 100))  # IoU percentage

    # Prepare input data for XGBoost
    if persons and vehicles:
        max_person = max(persons, key=lambda item: item[1])
        max_vehicle = max(vehicles, key=lambda item: item[1])
        xgb_input_data = [
            round(max_person[0][0], 5),
            round(max_person[0][1], 5),
            round(max_person[1], 5),
            round(max_vehicle[0][0], 5),
            round(max_vehicle[0][1], 5),
            round(max_vehicle[1], 5)
        ]

        # Run XGBoost classification
        xgb_prediction = xgb_model.predict([xgb_input_data])

        # Check the prediction and save the frame if condition is met
        if xgb_prediction[0] == 1:
            save_frame(video_frame[0])

    # Convert track_history to a tensor
    updated_track_history_tensor = np.array([pickle.dumps(track_history)], dtype=np.object_)
    
    return [updated_track_history_tensor]

# Helper functions
def calculate_line_equation(p1, p2):
    if p2[0] - p1[0] == 0:
        m = float('inf')
    else:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
    return m, b

def is_to_the_right(point, m, b):
    if m == float('inf'):
        return point[0] > b
    else:
        return point[1] > m * point[0] + b

def save_frame(video_frame):
    # Save the video frame to a directory on the server
    global frame_count
    frame_path = os.path.join(save_directory, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, video_frame)
    frame_count += 1
