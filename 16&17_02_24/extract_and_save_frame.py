import cv2
import csv
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

# Initialize tracking history and CSV writer
track_history = defaultdict(lambda: [])
csv_file = open('output_withbike.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Adjust the header for the CSV file to include only the maximum person and vehicle per frame
header = ['frame_number', 'person_center_x', 'person_center_y', 'person_velocity', 'person_intersection', 'vehicle_center_x', 'vehicle_center_y', 'vehicle_velocity', 'vehicle_intersection']
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
        center = np.mean(mask, axis=0)
        mask_binary = np.zeros_like(im0)
        cv2.fillPoly(mask_binary, [np.array(mask, dtype=np.int32)], color=(255, 255, 255))
        roi_binary = np.zeros_like(im0)
        cv2.fillPoly(roi_binary, [roi], color=(255, 255, 255))
        intersection_binary = cv2.bitwise_and(mask_binary, roi_binary)
        intersection_gray = cv2.cvtColor(intersection_binary, cv2.COLOR_BGR2GRAY)
        intersection = cv2.countNonZero(intersection_gray)

        if track_id in track_history:
            velocity = np.linalg.norm(center - track_history[track_id][-1]) if track_history[track_id] else 0
        else:
            velocity = 0
        track_history[track_id].append(center)

        if names[cls_num] in ['person'] and intersection > 0:
            persons.append((center, velocity, intersection))
        elif names[cls_num] in ['car', 'motorcycle', 'bus', 'truck'] and intersection > 0:
            vehicles.append((center, velocity, intersection))

        annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f'{track_id},{names[cls_num]}={confidence}')
        cv2.polylines(im0, [roi], True, (0, 255, 0), 2)

    # Ensure at least one person and one vehicle are detected within the ROI before proceeding
    if persons and vehicles:
        max_person = max(persons, key=lambda item: item[2])  # Find the person with the maximum intersection area
        max_vehicle = max(vehicles, key=lambda item: item[2])  # Find the vehicle with the maximum intersection area

        # Save only the data for the maximum person and vehicle to the CSV file
        data = [cap.get(cv2.CAP_PROP_POS_FRAMES), max_person[0][0], max_person[0][1], max_person[1], max_person[2], max_vehicle[0][0], max_vehicle[0][1], max_vehicle[1], max_vehicle[2]]
        csv_writer.writerow(data)

    cv2.imshow("instance-segmentation-object-tracking", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
