from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('crosswalk.pt')  # load a custom model

# Predict with the model
results = model('1.jpg', save=True)  # predict on an image