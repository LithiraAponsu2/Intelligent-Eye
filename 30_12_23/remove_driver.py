import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import numpy as np

track_history = defaultdict(lambda: [])

model = YOLO("yolov8x-seg.pt")
names = model.model.names

cap = cv2.VideoCapture("./test2.mp4")

out = cv2.VideoWriter('instance-segmentation-object-tracking2.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30, (int(cap.get(3)), int(cap.get(4))))

# Define lists to store identified bike riders and bike IDs with riders
bike_riders = set()
bike_with_riders = set()

pause = False

while True:
    if not pause:
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

        # Logic to track bike riders based on proximity with bicycles or motorcycles
        for i, (mask, track_id, confidence, cls_num) in enumerate(zip(masks, track_ids, confidences, cls_nums)):
            if names[cls_num] in ['bicycle', 'motorcycle']:
                vehicle_center = np.array([(mask[0] + mask[2]) / 2, (mask[1] + mask[3]) / 2])

                for j, (other_mask, other_track_id) in enumerate(zip(masks, track_ids)):
                    if j != i and names[cls_nums[j]] == 'person':
                        person_center = np.array([(other_mask[0] + other_mask[2]) / 2, (other_mask[1] + other_mask[3]) / 2])

                        distance_threshold = 50  # Adjust this threshold based on your scenario
                        distance = np.linalg.norm(vehicle_center - person_center)

                        # Check distance for 4 consecutive frames
                        if distance < distance_threshold:
                            track_history[other_track_id].append(track_id)
                            if len(track_history[other_track_id]) >= 4:
                                bike_riders.add(other_track_id)
                                bike_with_riders.add(track_id)

        for mask, track_id, confidence, cls_num in zip(masks, track_ids, confidences, cls_nums):
            annotator.seg_bbox(mask=mask,
                            mask_color=colors(track_id, True),
                            track_label=f'{track_id},{names[cls_num]}={confidence}')

        
        
        out.write(im0)
        cv2.imshow("instance-segmentation-object-tracking", im0)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):  # Press 'p' to pause/resume
            pause = not pause  # Toggle pause state

print("bike riders",bike_riders)
print("bike with riders", bike_with_riders)

out.release()
cap.release()
cv2.destroyAllWindows()
