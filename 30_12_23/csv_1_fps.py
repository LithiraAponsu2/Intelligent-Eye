import time
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

# Variables for FPS calculation
frame_count = 0
start_time = time.time()

# Initialize tracking history and CSV writer
track_history = defaultdict(lambda: [])
csv_file = open('output.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Define the header for the CSV file
header = ['frame_number']
for i in range(1, 5):
    header.extend([f'person_{i}_center_x', f'person_{i}_center_y', f'person_{i}_velocity', f'person_{i}_intersection'])
for i in range(1, 5):
    header.extend([f'vehicle_{i}_center_x', f'vehicle_{i}_center_y', f'vehicle_{i}_velocity', f'vehicle_{i}_intersection'])
csv_writer.writerow(header)

# Load the model
model = YOLO("yolov8x-seg.pt")
names = model.model.names

# Define the region of interest
roi = np.array([[508, 832], [608, 908], [1287, 831], [1132, 790]], np.int32)

# Open the video file
cap = cv2.VideoCapture("./test1.mp4")

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_count += 1        

    results = model.track(im0, persist=True)
    masks = results[0].masks.xy
    track_ids = results[0].boxes.id.int().cpu().tolist()
    confidences = results[0].boxes.conf.float().cpu().tolist()
    cls_nums = results[0].boxes.cls.int().cpu().tolist()

    annotator = Annotator(im0, line_width=2)

    # Initialize lists to store detected persons and vehicles
    persons = []
    vehicles = []

    for mask, track_id, confidence, cls_num in zip(masks, track_ids, confidences, cls_nums):
        # Calculate center of the mask
        center = np.mean(mask, axis=0)

        # Convert the mask to a binary image
        mask_binary = np.zeros_like(im0)
        cv2.fillPoly(mask_binary, [np.array(mask, dtype=np.int32)], color=(255, 255, 255))

        # Convert the ROI to a binary image
        roi_binary = np.zeros_like(im0)
        cv2.fillPoly(roi_binary, [roi], color=(255, 255, 255))

        # Calculate the intersection
        intersection_binary = cv2.bitwise_and(mask_binary, roi_binary)

        # Convert the intersection to grayscale
        intersection_gray = cv2.cvtColor(intersection_binary, cv2.COLOR_BGR2GRAY)

        # # Display the binary mask image
        # cv2.imshow(f'Mask {track_id}', mask_binary)

        # # Display the binary ROI image
        # cv2.imshow('ROI', roi_binary)        
        # break
        
        
        # Count the number of non-zero pixels in the intersection
        intersection = cv2.countNonZero(intersection_gray)
        # print(intersection)

        # Calculate velocity (relative to the previous position)
        if track_id in track_history:
            velocity = np.linalg.norm(center - track_history[track_id])
        else:
            velocity = 0
        track_history[track_id] = center

        # Store the detected person or vehicle
        if names[cls_num] in ['person'] and intersection > 0:
            persons.append((center, velocity, intersection))
        elif names[cls_num] in ['car', 'motorcycle', 'bus', 'truck'] and intersection > 0:
            vehicles.append((center, velocity, intersection))

        annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f'{track_id},{names[cls_num]}={confidence}')

        # Draw the ROI on the frame
        cv2.polylines(im0, [roi], True, (0, 255, 0), 2)
        
    # Write the data to the CSV file only if at least one person and one vehicle are detected inside the region
    if persons and vehicles:
        data = [cap.get(cv2.CAP_PROP_POS_FRAMES)]
        for i in range(4):
            if i < len(persons):
                data.extend([persons[i][0][0], persons[i][0][1], persons[i][1], persons[i][2]])
            else:
                data.extend([0, 0, 0, 0])
        for i in range(4):
            if i < len(vehicles):
                data.extend([vehicles[i][0][0], vehicles[i][0][1], vehicles[i][1], vehicles[i][2]])
            else:
                data.extend([0, 0, 0, 0])
        csv_writer.writerow(data)

    cv2.imshow("instance-segmentation-object-tracking", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate FPS
end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time
print(f"Processed {frame_count} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()
csv_file.close()
