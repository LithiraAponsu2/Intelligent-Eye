import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

# Initialize a default dictionary to keep track of objects
track_history = defaultdict(lambda: [])

# Load the YOLO model
model = YOLO("yolov8x-seg.pt")

# Get class names from the model
names = model.model.names

# Open the video file
cap = cv2.VideoCapture("show.mp4")

# Create a VideoWriter object to save the annotated video
out = cv2.VideoWriter('show-tracking.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30, (int(cap.get(3)), int(cap.get(4))))

while True:
    # Read a frame from the video
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Use the model to track objects in the frame
    results = model.track(im0, persist=True)
    masks = results[0].masks.xy
    track_ids = results[0].boxes.id.int().cpu().tolist()
    confidences = results[0].boxes.conf.float().cpu().tolist()
    cls_nums = results[0].boxes.cls.int().cpu().tolist()

    # Initialize the Annotator for drawing on the frame
    annotator = Annotator(im0, line_width=2)

    # Iterate over detected objects and annotate the frame
    for mask, track_id, confidence, cls_num in zip(masks, track_ids, confidences, cls_nums):
        if mask.size > 0:  # Check if the mask is not empty
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(track_id, True),
                               track_label=f'{track_id},{names[cls_num]}={confidence:.2f}')
        else:
            print(f"Empty mask for track ID {track_id} and class {names[cls_num]}")

    # Write the annotated frame to the output video
    out.write(annotator.result())

    # Display the frame
    cv2.imshow("instance-segmentation-object-tracking", annotator.result())

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter and VideoCapture objects and close all OpenCV windows
out.release()
cap.release()
cv2.destroyAllWindows()
