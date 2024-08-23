from flask import Flask, render_template, redirect, url_for, request, flash, session, Response
import glob
from flask import send_from_directory
from models import db, User
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import pickle
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db.init_app(app)

model = YOLO("yolov8x-seg.pt")
names = model.model.names
loaded_model = pickle.load(open("finalized_model_xg.sav", 'rb'))

file_name = 'show'
save_directory = "xgb_capt_2"
roi_file = 'roi_coordinates.npy'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

cap = cv2.VideoCapture(f"{file_name}.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Function to handle mouse events for selecting ROI
def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['coordinates'].append([x, y])
        print(f"Clicked at (x={x}, y={y})")
        cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select ROI", param['image'])
        if len(param['coordinates']) == 4:
            param['proceed'] = True

def calculate_line_equation(p1, p2):
    if p2[0] - p1[0] == 0:
        m = float('inf')
    else:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
    return m, b

def is_above_line(point, m, b):
    if m == float('inf'):
        return point[0] > p1[0]
    return point[1] > m * point[0] + b

def select_roi():
    callback_param = {'coordinates': [], 'proceed': False, 'image': None}
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        exit()
    callback_param['image'] = first_frame.copy()
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", get_mouse_coordinates, callback_param)
    print("Click on the first frame to mark 4 points for the ROI, then press any key to continue.")
    while not callback_param['proceed']:
        cv2.imshow("Select ROI", callback_param['image'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    if len(callback_param['coordinates']) != 4:
        print("Error: 4 points were not selected.")
        exit()
    roi = np.array(callback_param['coordinates'], np.int32)
    np.save(roi_file, roi)  # Save the ROI coordinates to a file
    return roi

# Load or select ROI
if os.path.exists(roi_file):
    roi = np.load(roi_file)
else:
    roi = select_roi()

roi_binary = np.zeros((1080, 1920, 3), dtype=np.uint8)
cv2.fillPoly(roi_binary, [roi], color=(255, 255, 255))
area_of_roi = cv2.countNonZero(cv2.cvtColor(roi_binary, cv2.COLOR_BGR2GRAY))
p1, p2 = roi[1], roi[2]
m, b = calculate_line_equation(p1, p2)

# Video streaming generator function
def generate_frames():
    track_history = defaultdict(lambda: [])
    persons_on_bikes = set()
    frame_count = 0
    while True:
        ret, im0 = cap.read()
        if not ret:
            break

        results = model.track(im0, persist=True)
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.float().cpu().tolist()
        cls_nums = results[0].boxes.cls.int().cpu().tolist()
        annotator = Annotator(im0, line_width=2)
        persons = []
        vehicles = []
        bike_vehicle_centers = []
        for mask, cls_num in zip(masks, cls_nums):
            if mask.size > 0 and names[cls_num] in ['bicycle', 'motorcycle']:
                center = np.mean(mask, axis=0)
                bike_vehicle_centers.append(center)

        for mask, track_id, confidence, cls_num in zip(masks, track_ids, confidences, cls_nums):
            if mask.size == 0:
                continue
            center = np.mean(mask, axis=0)
            if names[cls_num] == 'person' and any(np.linalg.norm(center - bike_center) < 100 for bike_center in bike_vehicle_centers):
                persons_on_bikes.add(track_id)
                continue
            mask_binary = np.zeros_like(im0)
            cv2.fillPoly(mask_binary, [np.array(mask, dtype=np.int32)], color=(255, 255, 255))
            intersection_binary = cv2.bitwise_and(mask_binary, roi_binary)
            intersection_area = cv2.countNonZero(cv2.cvtColor(intersection_binary, cv2.COLOR_BGR2GRAY))
            mask_area = cv2.countNonZero(cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY))
            iou = intersection_area / (mask_area + area_of_roi - intersection_area) if mask_area + area_of_roi - intersection_area > 0 else 0
            if track_id in track_history:
                velocity = np.linalg.norm(center - track_history[track_id][-1]) if track_history[track_id] else 0
            else:
                velocity = 0
            track_history[track_id].append(center)
            if names[cls_num] == 'person' and intersection_area > 0 and track_id not in persons_on_bikes:
                persons.append((center, velocity, iou * 100))
            elif names[cls_num] in ['car', 'motorcycle', 'bus', 'truck'] and intersection_area > 0:
                max_point = max(mask, key=lambda x: (x[0], x[1]))
                if not is_above_line(max_point, m, b):
                    vehicles.append((center, velocity, iou * 100))
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f'{track_id},{names[cls_num]}={confidence}')
            cv2.polylines(im0, [roi], True, (0, 255, 0), 2)

        if persons and vehicles:
            max_person = max(persons, key=lambda item: item[2])
            max_vehicle = max(vehicles, key=lambda item: item[2])
            data = [
                round(max_person[0][0], 5),
                round(max_person[0][1], 5),
                round(max_person[1], 5),
                round(max_person[2], 5),
                round(max_vehicle[0][0], 5),
                round(max_vehicle[0][1], 5),
                round(max_vehicle[1], 5),
                round(max_vehicle[2], 5)
            ]
            prediction = loaded_model.predict([data])
            if prediction[0] == 1:
                frame_path = os.path.join(save_directory, f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg")
                cv2.imwrite(frame_path, im0)
                print(f"Saved frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} due to prediction 1.")
        
        _, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def home():
    return render_template('signin.html')

@app.route('/create-account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already taken. Please choose another username.')
            return redirect(url_for('create_account'))
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully!')
        return redirect(url_for('home'))
    return render_template('create_account.html')

@app.route('/signin', methods=['POST'])
def signin():
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        session['user_id'] = user.id
        flash('Signed in successfully')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid email or password')
        return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html')

@app.route('/tab_2')
def reports():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('tab_2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violation_frames')
def violation_frames():
    image_paths = sorted(glob.glob(os.path.join(save_directory, "*.jpg")), key=os.path.getmtime, reverse=True)
    images = [os.path.basename(path) for path in image_paths]
    return {'images': images}

@app.route('/xgb_capt_2/<filename>')
def uploaded_file(filename):
    return send_from_directory(save_directory, filename)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
