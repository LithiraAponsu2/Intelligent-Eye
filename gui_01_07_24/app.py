from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import pickle
import os
import shutil

app = Flask(__name__, static_folder='images')

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
def is_above_line(point, m, b):
    # Check for vertical line
    if m == float('inf'):
        return point[0] > p1[0]  # Assuming p1[0] is the x-coordinate of the vertical line
    return point[1] > m * point[0] + b

# Initialize variables
file_name = 'show'
save_directory = "images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load the YOLO model
model = YOLO("yolov8x-seg.pt")
names = model.model.names

# Load your pre-trained model (adjust the path as necessary)
loaded_model = pickle.load(open("finalized_model_xg.sav", 'rb'))

# Initialize video capture
cap = cv2.VideoCapture('show.mp4')

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
m, b = calculate_line_equation(p1, p2)

# Initialize tracking history
track_history = defaultdict(lambda: [])

# Define a set to track IDs of persons on bikes
persons_on_bikes = set()

# Function to process each frame
def process_frame(frame):
    results = model.track(frame, persist=True)
    masks = results[0].masks.xy
    track_ids = results[0].boxes.id.int().cpu().tolist()
    confidences = results[0].boxes.conf.float().cpu().tolist()
    cls_nums = results[0].boxes.cls.int().cpu().tolist()

    annotator = Annotator(frame, line_width=2)

    # Initialize lists to store detected persons and vehicles
    persons = []
    vehicles = []

    # Identify all persons and bikes/motorcycles
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
        if names[cls_num] == 'person' and any(np.linalg.norm(center - bike_center) < 100 for bike_center in bike_vehicle_centers):
            persons_on_bikes.add(track_id)
            continue  # Skip further processing for this person

        mask_binary = np.zeros_like(frame)
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
            max_point = max(mask, key=lambda x: (x[0], x[1]))
            if not is_above_line(max_point, m, b):
                vehicles.append((center, velocity, iou * 100))  # iou percentage
        
        annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f'{track_id},{names[cls_num]}={confidence}')
        cv2.polylines(frame, [roi], True, (0, 255, 0), 2)

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
        
        # Assuming your model expects a 2D array for a single sample
        prediction = loaded_model.predict([data])
        
        # Check the prediction and save the frame if condition is met (e.g., prediction is 1)
        if prediction[0] == 1:
            frame_path = os.path.join(save_directory, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} due to prediction 1.")

    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/default_thumbnails')
def default_thumbnails():
    image_dir = 'images'  # Directory where images are stored
    image_paths = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_path = os.path.join(image_dir, filename)
            image_paths.append(image_path)
    return jsonify(image_paths)

app.static_folder = 'images'

@app.route('/image_filenames')
def image_filenames():
    image_dir = os.path.join(os.path.dirname(__file__), 'images')  # Image directory path relative to app.py
    image_filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    image_paths = [os.path.join(filename) for filename in image_filenames]  # Full paths relative to the static folder
    return jsonify(image_paths)

@app.route('/save_selected_frames', methods=['POST'])
def save_selected_frames():
    data = request.json
    selected_images = data.get('images', [])
    
    if not selected_images:
        return jsonify({'message': 'No frames selected!'}), 400

    new_folder = 'selected_frames'
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for image in selected_images:
        source_path = os.path.join('images', image)
        destination_path = os.path.join(new_folder, image)
        shutil.copyfile(source_path, destination_path)

    return jsonify({'message': f'Successfully saved {len(selected_images)} frames to {new_folder}.'})

if __name__ == "__main__":
    app.run(debug=True)
