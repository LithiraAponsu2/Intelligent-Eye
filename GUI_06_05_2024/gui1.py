import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QImage, QPixmap
import cv2

class TrafficViolationApp(QMainWindow):
    def __init__(self, video_source):
        super().__init__()

        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.video_label = QLabel()
        self.frame_label = QLabel()
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.frame_label)
        
        self.central_widget.setLayout(layout)
        
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_video)
        
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_video)
        
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        self.update()
        
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
            
            # Assuming you have a captured frame stored in `captured_frame`
            # Replace `captured_frame` with your actual captured frame
            captured_frame = frame  # Placeholder, replace with actual captured frame
            q_img_captured = QImage(captured_frame.data, captured_frame.shape[1], captured_frame.shape[0], captured_frame.strides[0], QImage.Format_RGB888)
            pixmap_captured = QPixmap.fromImage(q_img_captured).scaledToWidth(400)
            self.frame_label.setPixmap(pixmap_captured)
        
        # Update every 10 ms
        QtCore.QTimer.singleShot(10, self.update)
        
    def start_video(self):
        # Start video capture
        self.vid = cv2.VideoCapture(self.video_source)
        
    def stop_video(self):
        # Release video capture
        if self.vid.isOpened():
            self.vid.release()

# Create a Qt application
app = QApplication(sys.argv)
window = TrafficViolationApp("your_video_source.mp4")
window.show()
sys.exit(app.exec_())
