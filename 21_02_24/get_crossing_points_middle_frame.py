import cv2
import numpy as np

# Function to handle mouse events
def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append([x, y])  # Store each coordinate as a list within the coordinates list
        print(f"Clicked at (x={x}, y={y})")
        # Proceed to draw after collecting 4 points
        if len(coordinates) == 4:
            param['proceed'] = True  # Signal to proceed with drawing

# Open the video file
video_path = "4.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the middle frame
middle_frame = total_frames // 3

coordinates = []  # List to store the clicked coordinates in nested list format
cv2.namedWindow("Video")

# Parameters for control flow
callback_param = {'proceed': False}

cv2.setMouseCallback("Video", get_mouse_coordinates, callback_param)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or video file not found.")
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Display and capture points on the middle frame
    if current_frame == middle_frame:
        print("Middle frame reached. Click on the frame to mark 4 points, then wait.")
        while not callback_param['proceed']:
            cv2.imshow("Video", frame)
            cv2.waitKey(1)
    
    # Once points are captured, move to the next frame to draw
    if callback_param['proceed'] and current_frame == middle_frame + 1:
        if len(coordinates) >= 3:  # Ensure there are enough points for a contour
            pts = np.array(coordinates, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # Draw the contour on the frame
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.imshow("Video", frame)
        print("Contour drawn. Press any key to exit.")
        cv2.waitKey(0)  # Wait for key press to exit
        break  # Exit the loop after showing the contour

# Print the coordinates in the desired nested list format
print("Coordinates used for drawing:", coordinates)

cap.release()
cv2.destroyAllWindows()
