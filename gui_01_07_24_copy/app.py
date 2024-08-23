from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import os
import base64
import shutil
from datetime import datetime

app = Flask(__name__)

# Path for saving images
save_directory = "/home/lithira/saved_frames/client_1"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Global variable to store the path of the video file
video_path = "received_video.mp4"

def decode_and_save_video(video_base64, save_path):
    """Decode base64 video and save it as a file."""
    video_bytes = base64.b64decode(video_base64)
    with open(save_path, 'wb') as f:
        f.write(video_bytes)

def generate_frames():
    """Generator function to read and yield video frames."""
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video if it ends
            continue
        else:
            # Encode the frame as a JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to stream video frames to the client."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/receive_video', methods=['POST'])
def receive_video():
    """Route to receive base64-encoded video."""
    data = request.json
    video_base64 = data.get('video', '')
    
    if not video_base64:
        return jsonify({'message': 'No video data received!'}), 400

    decode_and_save_video(video_base64, video_path)

    return jsonify({'message': 'Video successfully received and saved.'})

@app.route('/image_filenames')
def image_filenames():
    """Route to get the filenames of saved images."""
    image_filenames = [filename for filename in os.listdir(save_directory) if filename.endswith(('.jpg', '.jpeg', '.png'))]
    return jsonify(image_filenames)

@app.route('/images/<filename>')
def send_image(filename):
    """Route to serve an image from the save directory."""
    return send_from_directory(save_directory, filename)

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Route to capture the current video frame and save it as an image."""
    frame_file = request.files.get('frame')
    if frame_file:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f"frame_{timestamp}.jpg"
        filepath = os.path.join(save_directory, filename)
        frame_file.save(filepath)
        return jsonify({'message': f'Frame captured and saved as {filename}.'})
    else:
        return jsonify({'message': 'Failed to capture frame.'}), 400

@app.route('/save_selected_frames', methods=['POST'])
def save_selected_frames():
    """Route to save selected frames to a separate directory."""
    data = request.json
    selected_images = data.get('images', [])
    
    if not selected_images:
        return jsonify({'message': 'No frames selected!'}), 400

    new_folder = 'selected_frames'
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for image in selected_images:
        source_path = os.path.join(save_directory, image)
        destination_path = os.path.join(new_folder, image)
        shutil.copyfile(source_path, destination_path)

    return jsonify({'message': f'Successfully saved {len(selected_images)} frames to {new_folder}.'})

if __name__ == "__main__":
    app.run(debug=True)
