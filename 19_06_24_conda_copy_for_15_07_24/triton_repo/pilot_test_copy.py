import cv2
import time

url = "https://192.168.8.124:8080/video"  # Adjust as needed
max_retries = 5
retry_delay = 2

for attempt in range(max_retries):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print(f"Successfully connected on attempt {attempt + 1}")
        break
    else:
        print(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
else:
    print(f"Failed to connect after {max_retries} attempts. Please check your setup.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame. Trying to reconnect...")
        cap.release()
        cap = cv2.VideoCapture(url)
        continue
    
    cv2.imshow("IV Cam Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()