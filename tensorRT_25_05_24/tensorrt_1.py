import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import os
import pickle
import time
import csv

"""
Point Selection

1=============4
 \             \
  \             \       
   \             \
    2=============3
    
"""

# Function to handle mouse events for selecting ROI
def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['coordinates'].append([x, y])  # Store each coordinate as a list within the coordinates list
        print(f"Clicked at (x={x}, y={y})")
        cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)  # Visual feedback for point selection
        cv2.imshow("Select ROI", param['image'])  # Update the image with the drawn point
        if len(param['coordinates']) == 4:
            param['proceed'] = True  # Signal to proceed with drawing after collecting 4 points

# Calculate slope and y-intercept of line
def calculate_line_equation(p1, p2):
    if p2[0] - p1[0] == 0:
        m = float('inf')
    else:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
    return m, b

# Check if point is below the line
def is_to_the_right(point, m, b):
    if m == float('inf'):
        # For a vertical line, check if the point's x-coordinate is greater than the line's x-coordinate
        return point[0] > b
    else:
        # For a non-vertical line, check if the point's y-coordinate is above the line
        return point[1] > m * point[0] + b

# Initialize variables
file_name = 'show'
save_directory = "xgb_capt_3"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load the YOLO model
# model = YOLO("yolov8x-seg.pt")
tensorrt_model = YOLO("yolov8x-seg.engine")
names = tensorrt_model.names

# Load your pre-trained model (adjust the path as necessary)
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

# Calculate line equation from ROI points 1 and 2
p1, p2 = roi[1], roi[2]
print(p1, p2)
m, b = calculate_line_equation(p1, p2)
print(m,b)

# Initialize tracking history
track_history = defaultdict(lambda: [])

# Define a set to track IDs of persons on bikes
persons_on_bikes = set()

# Performance Analysis Metrics
yolo_processing_times = []
model_inference_times = []
frame_processing_times = []

# Create and open CSV file for logging performance metrics
with open('performance_metrics3.csv', 'w', newline='') as csvfile:
    fieldnames = ['Frame', 'YOLO Processing Time (s)', 'Model Inference Time (s)', 'Frame Processing Time (s)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Process video frames
    frame_count = 0
    while True:
        start_time_frame = time.time()
        ret, im0 = cap.read()
        if not ret:
            break  # Exit loop if video frame is empty

        # YOLO Object Detection
        start_time_yolo = time.time() #########
        results = tensorrt_model.track(im0, persist=True, classes=[0,2,3,5,7])
        
        # Model Inference
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.float().cpu().tolist()
        cls_nums = results[0].boxes.cls.int().cpu().tolist()

        annotator = Annotator(im0, line_width=2)

        end_time_yolo = time.time() #########
        yolo_processing_time = end_time_yolo - start_time_yolo ########
        yolo_processing_times.append(yolo_processing_time) #########
        
        
        start_time_inference = time.time() #########
        # Initialize lists to store detected persons and vehicles
        persons = []
        vehicles = []

        # Identify all persons and bikes/motorcycles
        bike_vehicle_centers = []
        for mask, cls_num in zip(masks, cls_nums):
            if mask.size > 0 and names[cls_num] in ['motorcycle']:
                center = np.mean(mask, axis=0)
                bike_vehicle_centers.append(center)

        # data extract
        for mask, track_id, confidence, cls_num in zip(masks, track_ids, confidences, cls_nums):
            if mask.size == 0:
                continue  # Skip if mask is empty
            
            max_point = max(mask, key=lambda x: (x[0], x[1]))
            if is_to_the_right(max_point, m, b):
                frame_count-=1
                continue  # skip vehicle passed the crossing upper bound
            
            center = np.mean(mask, axis=0)

            # Check if this person is on a bike
            if names[cls_num] == 'person' and any(np.linalg.norm(center - bike_center) < 100 for bike_center in bike_vehicle_centers):
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

            if names[cls_num] == 'person' and intersection_area > 0 and track_id not in persons_on_bikes:
                persons.append((center, velocity, iou * 100))  # iou percentage
            elif names[cls_num] in ['car', 'motorcycle', 'bus', 'truck'] and intersection_area > 0:
                vehicles.append((center, velocity, iou * 100))  # iou percentage

            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f'{track_id},{names[cls_num]}={confidence}')
            cv2.polylines(im0, [roi], True, (0, 255, 0), 2)

        # Model Inference Time Calculation
        end_time_inference = time.time() ###########
        model_inference_time = end_time_inference - start_time_inference #############
        model_inference_times.append(model_inference_time) ##############

        # Frame Processing Time Calculation
        end_time_frame = time.time() ################
        frame_processing_time = end_time_frame - start_time_frame #########
        frame_processing_times.append(frame_processing_time) ###########

        # Write performance metrics to CSV
        writer.writerow({'Frame': frame_count,
                         'YOLO Processing Time (s)': yolo_processing_time,
                         'Model Inference Time (s)': model_inference_time,
                         'Frame Processing Time (s)': frame_processing_time})

        # Continue with prediction and frame saving logic here, now with the updated 'vehicles' list
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

            prediction = loaded_model.predict([data])

            # Check the prediction and save the frame if condition is met (e.g., prediction is 1)
            if prediction[0] == 1:
                frame_path = os.path.join(save_directory, f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg")
                cv2.imwrite(frame_path, im0)

        cv2.imshow("instance-segmentation-object-tracking", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

# Calculate and print average processing times
avg_yolo_processing_time = sum(yolo_processing_times) / len(yolo_processing_times)
avg_model_inference_time = sum(model_inference_times) / len(model_inference_times)
avg_frame_processing_time = sum(frame_processing_times) / len(frame_processing_times)

print(f"Average YOLO Processing Time: {avg_yolo_processing_time} seconds")
print(f"Average Model Inference Time: {avg_model_inference_time} seconds")
print(f"Average Frame Processing Time: {avg_frame_processing_time} seconds")

print(f"Full YOLO Processing Time: {sum(yolo_processing_times)} seconds")
print(f"Full Model Inference Time: {sum(model_inference_times)} seconds")
print(f"Full Frame Processing Time: {sum(frame_processing_times)} seconds")

print(f'Total Frames: {frame_count}')


# Close the video capture object and OpenCV windows
cap.release()
cv2.destroyAllWindows()
