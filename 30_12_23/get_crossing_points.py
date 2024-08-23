import cv2

# Function to handle mouse events
def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        print(f"Clicked at (x={x}, y={y})")

# Open the video file
video_path = "test2.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

coordinates = []  # List to store the clicked coordinates
cv2.namedWindow("Video")

cv2.setMouseCallback("Video", get_mouse_coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or video file not found.")
        break

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print("Captured coordinates:", coordinates)
cap.release()
cv2.destroyAllWindows()
