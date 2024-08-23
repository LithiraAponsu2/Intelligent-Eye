import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import joblib
import os
import pickle

# Function to handle mouse events for selecting ROI
def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['coordinates'].append([x, y])  # Store each coordinate as a list within the coordinates list
        print(f"Clicked at (x={x}, y={y})")
        cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)  # Visual feedback for point selection
        cv2.imshow("Select ROI", param['image'])  # Update the image with the drawn point
        if len(param['coordinates']) == 4:
            param['proceed'] = True  # Signal to proceed with drawing after collecting 4 points

# Initialize variables
file_name = '4'
save_directory = "xgb_frames"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load the YOLO model
model = YOLO("yolov8x-seg.pt")
names = model.model.names

# Load your pre-trained model (adjust the path as necessary)
# random_forest_model = joblib.load("random_forest_model.joblib")

# Example of loading the model back into memory
loaded_model = pickle.load(open("finalized_model_xg.sav", 'rb'))

# Open the video file
cap = cv2.VideoCapture(f"{file_name}.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Parameters for control flow and mouse callback
callback_param = {'coordinates': [], 'proceed': False, 'image': None}

# Read first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

callback_param['image'] = first_frame.copy()

# Setup mouse callback to capture ROI
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", get_mouse_coordinates, callback_param)

print("Click on the first frame to mark 4 points for the ROI, then press any key to continue.")

# Show the first frame and wait for the ROI to be selected
while not callback_param['proceed']:
    cv2.imshow("Select ROI", callback_param['image'])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Ensure 4 points were selected for the ROI
if len(callback_param['coordinates']) != 4:
    print("Error: 4 points were not selected.")
    exit()

# Define the region of interest with the selected points
roi = np.array(callback_param['coordinates'], np.int32)
roi_binary = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Adjust the size if necessary
cv2.fillPoly(roi_binary, [roi], color=(255, 255, 255))
area_of_roi = cv2.countNonZero(cv2.cvtColor(roi_binary, cv2.COLOR_BGR2GRAY))  # Calculate area of ROI

# Initialize tracking history
track_history = defaultdict(lambda: [])

# Define a set to track IDs of persons on bikes
persons_on_bikes = set()

# Process video frames
while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.track(im0, persist=True)
    masks = results[0].masks.xy
    track_ids = results[0].boxes.id.int().cpu().tolist()
    confidences = results[0].boxes.conf.float().cpu().tolist()
    cls_nums = results[0].boxes.cls.int().cpu().tolist()

    annotator = Annotator(im0, line_width=2)

    # Initialize lists to store detected persons and vehicles
    persons = []
    vehicles = []

    # Identify all persons and bikes/motorcycles to find persons on bikes
    bike_vehicle_centers = []
    for mask, cls_num in zip(masks, cls_nums):
        if mask.size > 0 and names[cls_num] in ['bicycle', 'motorcycle']:
            center = np.mean(mask, axis=0)
            bike_vehicle_centers.append(center)

    for mask, track_id, confidence, cls_num in zip(masks, track_ids, confidences, cls_nums):
        if mask.size == 0:
            continue  # Skip if mask is empty

        center = np.mean(mask, axis=0)

        # Check if this person is on a bike
        if names[cls_num] == 'person':
            if any(np.linalg.norm(center - bike_center) < 100 for bike_center in bike_vehicle_centers):
                persons_on_bikes.add(track_id)
                continue  # Skip further processing for this person

        mask_binary = np.zeros_like(im0)
        cv2.fillPoly(mask_binary, [np.array(mask, dtype=np.int32)], color=(255, 255, 255))
        intersection_binary = cv2.bitwise_and(mask_binary, roi_binary)
        intersection_area = cv2.countNonZero(cv2.cvtColor(intersection_binary, cv2.COLOR_BGR2GRAY))
        mask_area = cv2.countNonZero(cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY))

        # Calculate IoU
        iou = intersection_area / (mask_area + area_of_roi - intersection_area) if mask_area + area_of_roi - intersection_area > 0 else 0

        if track_id in track_history:
            velocity = np.linalg.norm(center - track_history[track_id][-1]) if track_history[track_id] else 0
        else:
            velocity = 0
        track_history[track_id].append(center)

        if names[cls_num] in ['person'] and intersection_area > 0 and track_id not in persons_on_bikes:
            persons.append((center, velocity, iou * 100))  # iou percentage
        elif names[cls_num] in ['car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'] and intersection_area > 0:
            vehicles.append((center, velocity, iou * 100))  # iou percentage

        annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f'{track_id},{names[cls_num]}={confidence}')
        cv2.polylines(im0, [roi], True, (0, 255, 0), 2)

    # Prediction and frame saving logic (if needed) goes here
    # (The rest of the first script's logic continues...)
        # Save data to CSV for the maximum intersection person and vehicle
    if persons and vehicles:
        max_person = max(persons, key=lambda item: item[2])
        max_vehicle = max(vehicles, key=lambda item: item[2])
        # Round values to 5 decimal points before writing to CSV
        data = [
            # round(cap.get(cv2.CAP_PROP_POS_FRAMES), 5),
            round(max_person[0][0], 5),
            round(max_person[0][1], 5),
            round(max_person[1], 5),
            round(max_person[2], 5),  # IoU percentage for person
            round(max_vehicle[0][0], 5),
            round(max_vehicle[0][1], 5),
            round(max_vehicle[1], 5),
            round(max_vehicle[2], 5)  # IoU percentage for vehicle
        ]
        
        # Assuming your model expects a 2D array for a single sample
        # prediction = random_forest_model.predict([data])
        
        prediction = loaded_model.predict([data])
        
        
        
        # Check the prediction and save the frame if condition is met (e.g., prediction is 1)
        if prediction[0] == 1:
            frame_path = os.path.join(save_directory, f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg")
            cv2.imwrite(frame_path, im0)
            print(f"Saved frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} due to prediction 1.")

    cv2.imshow("instance-segmentation-object-tracking", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
